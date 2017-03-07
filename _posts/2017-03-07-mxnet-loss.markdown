---
layout: post
title: "mxnet loss"
date: 2017-03-07
mathjax: true
categories: code
tags: mxnet loss
---
* content
{:toc}
本文写作时候2017.3.7，mxnet loss有点晦涩，写作本文梳理一下。mxnet讨论区说近期会重构loss layer. 待重构后再做更新。



本文介绍mxnet loss，反向传播，自定义损失函数，metric. 先从mxnet center loss作为例子，逐渐扩大讲解范围。



### mxnet center loss

#### 自定义op

自定义op如下。具体参考[这里](http://mxnet.io/tutorials/python/symbol.html#customized-symbol)

```python
class CenterLoss(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, num_class, alpha, scale=1.0):
        if not len(shapes[0]) == 2:
            raise ValueError('dim for input_data shoudl be 2 for CenterLoss')   
        self.alpha = alpha
        self.batch_size = shapes[0][0]
        self.num_class = num_class
        self.scale = scale

    def forward(self, is_train, req, in_data, out_data, aux):
        labels = in_data[1].asnumpy()
        diff = aux[0]
        center = aux[1]
        # store x_i - c_yi
        for i in range(self.batch_size):
            diff[i] = in_data[0][i] - center[int(labels[i])]
        loss = mx.nd.sum(mx.nd.square(diff)) / self.batch_size / 2
        self.assign(out_data[0], req[0], loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        diff = aux[0]
        center = aux[1]
        sum_ = aux[2]
        # back grad is just scale * ( x_i - c_yi)
        grad_scale = float(self.scale)
        self.assign(in_grad[0], req[0], diff * grad_scale)
        # update the center
        labels = in_data[1].asnumpy()
        label_occur = dict()
        for i, label in enumerate(labels):
            label_occur.setdefault(int(label), []).append(i)
        for label, sample_index in label_occur.items():
            sum_[:] = 0
            for i in sample_index:
                sum_ = sum_ + diff[i]
            delta_c = sum_ / (1 + len(sample_index))
            center[label] += self.alpha * delta_c
```

在forward里面，根据in_data计算out_data;

在backward中，根据输入数据in_data, 输出梯度out_grad来计算后向梯度in_grad

这里forward, backward都使用了assign函数: 

```python
def assign(self, dst, req, src):
    """Helper function for assigning into dst depending on requirements."""
    if req == 'null':
        return
    elif req == 'write' or req == 'inplace':
        dst[:] = src
    elif req == 'add':
        dst[:] += src
```

其实是根据req的操作类型了来将src拷贝到dst.



使用aux 0,1,2分别保存计算center loss 所需要的diff, center, sum.

前向时候记录diff, 后向时候累积diff的sum，并更新label的center。**这里的sum其实不要使用aux吧？**





设计要点见原[博客](https://pangyupo.github.io/2016/10/16/mxnet-center-loss/)

> - centerloss这个operator有个稍微特殊一点的地方，它内含的各个类别的中心center，既不是`in_data`也不是`out_data`，又不是普通的weight(它并不是随着grad去更新的，有自己的更新手段)，所以最终把这个作为了`auxiliary`。但是对多卡来说，这里其实不是很准确，由于mxnet采用的数据划分，所以在每个卡上都有一套完整的参数。可是centerloss的参数center是aux变量，所以会各自有一套不同的center。暂时没有想到更好的方式来处理这个问题，不过由于特征抽取的网络是同步的，所以每个center差别也不大，最好的训练结果其实没有太大的区别，当然我只在MNIST这个小库上对比了一发，如果你有更好的方法请告诉我 :)
> - 最后，由于使用了aux变量，需要添加`list_auxiliary_states`函数表明需求， 并在`infer_shape`函数中返回正确的大小。每个参数都加上了bias字样，因为mxnet会根据不同的名字选择不同的初始化方式，bias一般都使用零初始化。另外一个比较坑的地方是，`diff_bias`的shape是需要和`in_data`一致，但是不要使用`in_shape`来推断。假如每个batch的大小的是100，那么`diff_bias`应该是(100,2)的shape，但是如果我有两个gpu卡，在每个卡在下一次`infer_shape`的时候会得到(50,2)。由于mxnet是先申请一次cpu的内存，然后把参数复制过去，这里会因为shape不一致而报错。所以这里才用了直接计算每张卡的样本数目，当作参数传递进去。



自定义op还需要一下定式化代码, 用于描述op输入，输出，shape信息。

```python
@mx.operator.register("centerloss")
class CenterLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_class, alpha, scale=1.0, batchsize=64):
        super(CenterLossProp, self).__init__(need_top_grad=False)
        # convert it to numbers
        self.num_class = int(num_class)
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.batchsize = int(batchsize)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        # call them 'bias' for zero initialization
        return ['diff_bias', 'center_bias', 'sum_bias']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        # store diff , same shape as input batch
        diff_shape = [self.batchsize, data_shape[1]]
        # store the center of each class , should be ( num_class, d )
        center_shape = [self.num_class, diff_shape[1]]
        # computation buf
        sum_shape = [diff_shape[1],]
        output_shape = [1, ]
        return [data_shape, label_shape], [output_shape], [diff_shape, center_shape, sum_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CenterLoss(ctx, shapes, dtypes, self.num_class, self.alpha, self.scale)
```



#### 网络构造

```python
def get_symbol(batchsize=64):
    data = mx.symbol.Variable('data')
    softmax_label = mx.symbol.Variable('softmax_label')
    center_label = mx.symbol.Variable('center_label')
	# ...略去部分
    fc2 = mx.symbol.FullyConnected(data=embedding, num_hidden=10, name='fc2')
    ce_loss = mx.symbol.SoftmaxOutput(data=fc2, label=softmax_label, name='softmax')
    center_loss_ = mx.symbol.Custom(data=fc2, label=center_label, name='center_loss_', op_type='centerloss',\
            num_class=10, alpha=0.5, scale=0.01, batchsize=batchsize)
    center_loss = mx.symbol.MakeLoss(name='center_loss', data=center_loss_)
    mlp = mx.symbol.Group([ce_loss, center_loss])
    return mlp
```

这里可以看到将softmax的损失函数和center_loss的损失函数Group在一起，优化这个目标函数。

同一个label被同时用于softmax_label和center_label, 需要改造data loader:

```python
 @property
 def provide_label(self):
   provide_label = self.data_iter.provide_label[0]
   return [('softmax_label', provide_label[1]), \
   ('center_label', provide_label[1])]
```

#### metric

为了随时观察损失函数的值，定义metric:

```python
class Accuracy(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(Accuracy, self).__init__('accuracy', num)

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        if self.num is not None:
            assert len(labels) == self.num
        pred_label = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')
        mx.metric.check_label_shapes(label, pred_label)
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

# define some metric of center_loss
class CenterLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(CenterLossMetric, self).__init__('center_loss')

    def update(self, labels, preds):
        self.sum_metric = + preds[1].asnumpy()[0]
        self.num_inst = 1
```

合并metric:

```python
eval_metrics = mx.metric.CompositeEvalMetric()
eval_metrics.add(Accuracy())
eval_metrics.add(CenterLossMetric())
```



### Group

合并多个损失函数。

```python
def get_loss(gram, content):
  gram_loss = []
  for i in range(len(gram.list_outputs())):
      gvar = mx.sym.Variable("target_gram_%d" % i)
      gram_loss.append(mx.sym.sum(mx.sym.square(gvar - gram[i])))
  cvar = mx.sym.Variable("target_content")
  content_loss = mx.sym.sum(mx.sym.square(cvar - content))
  return mx.sym.Group(gram_loss), content_loss
```


### MakeLoss

```python
ce_loss = mx.symbol.SoftmaxOutput(data=fc2, label=softmax_label, name='softmax')
center_loss_ = mx.symbol.Custom(data=fc2, label=center_label, name='center_loss_', op_type='centerloss',\
        num_class=10, alpha=0.5, scale=0.01, batchsize=batchsize)
center_loss = mx.symbol.MakeLoss(name='center_loss', data=center_loss_)
mlp = mx.symbol.Group([ce_loss, center_loss])
```


### output layer

- SoftmaxOutput

  $$out[i,:] = softmax(data[i,:])$$

  $$softmax(x) = \left[..., \frac{exp(x[j])}{exp(x[0])+...+exp(x[k-1])}, ...\right]$$

  Softmax with logit loss.In the forward pass, the softmax output is returned. In the backward pass, the logit loss, also called cross-entroy loss, is added.**

- LinearRegressionOutput

- LogisticRegressionOutput

- MAERegressionOutput

- SVMOutput

使用各种output layer可以不指定参数直接用backward()计算梯度, 而如果是neural style这种不是output layer的要手动将gram_diff, content_diff backward回去

```python
gram_executors[j].forward(is_train=True)
gram_diff[j] = gram_executors[j].outputs[0]-target_grams[j]
gram_executors[j].backward(gram_diff[j])
gram_grad[j][i:i+1] = gram_executors[j].grad_dict['gram_data'] / batch_size
```

```python
content_grad = alpha*(desc_executor.outputs[len(style_layer)]-target_content) / layer_size / batch_size #注意: 手动计算梯度
desc_executor.backward(gram_grad+[content_grad])
gene_executor.backward(desc_executor.grad_dict['data']+tv_grad_executor.outputs[0])
for i, var in enumerate(gene_executor.grad_dict):
    if var != 'data':
        optimizer.update(i, gene_executor.arg_dict[var], gene_executor.grad_dict[var], optim_states[i])
```

backward是计算梯度，update是真正根据梯度更新参数。更新策略在optimizer里面

### metric输出

如center loss的metric所示，loss的计算与metric的输出是独立的。两者需要分别实现。

对于使用LogisticRegressionOutput等loss layer, 在forward之后可以马上backward, 无需管是否update_metric.

metric输出并不是必须的，也可以手动进行输出

```python
loss[len(style_layer)] += alpha*np.sum(np.square((desc_executor.outputs[len(style_layer)]-target_content).asnumpy()/np.sqrt(layer_size))) / batch_size　#注意:手动计算loss，不用metric
if epoch % 20 == 0:
	print 'loss', sum(loss), np.array(loss)
```



### L1, L2 norm

#### smooth_l1:

```python
cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
# bounding box regression
bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')
group = mx.symbol.Group([cls_prob, bbox_loss])
```

从上面可以看出，有多个损失函数可以用Group合并起来。

#### L2

l2正则化通过weight decay实现：

先通过backward算出纯梯度，再通过一下公式更新正则化的梯度。`个人看法...`

```
weight[:] += -lr * (grad + weight_decay * weight)
```



