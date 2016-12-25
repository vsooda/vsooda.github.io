---
layout: post
title: "mxnet optimizer"
date: 2016-12-04
categories: code
tags: mxnet optimizer
---
* content
{:toc}
### mxnet参数优化 

lr-schedure 包含在optimizer之内

在init_optimizer的时候，get_updater

```python
def get_updater(optimizer):
    states = dict()
    def updater(index, grad, weight):
        if index not in states:
            states[index] = optimizer.create_state(index, weight)
        optimizer.update(index, weight, grad, states[index])
    return updater
```

训练的时候：

```python
for nbatch, data_batch in enumerate(train_dataiter):
    mod.forward(data_batch)
    mod.update_metric(metric, data_batch.label) #计算误差
    mod.backward() #计算当前批次的梯度
    mod.update() #更新参数，因为当前梯度只是少量样本的梯度，所以不能完全按照梯度更新。需要设置学习率等
```

* forward
* backward
* update
    * updater(index,grad, weight) 里面调用optimizer的update函数

每个optimizer需要实现的是__init__, create_state, update。 以SGD为例子：

```python
class SGD(Optimizer):
    def __init__(self, momentum=0.0, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        if self.momentum == 0.0:
            return None
        else:
            return zeros(weight.shape, weight.context, dtype=weight.dtype)

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state #注意，这里并没有复制。更改mom会使得state也产生，变化。达到记录上一次更新的目的。
            mom[:] *= self.momentum
            mom[:] += -lr * (grad + wd * weight)
            weight[:] += mom
        else:
            assert self.momentum == 0.0
            weight[:] += -lr * (grad + wd * weight)
```



### finetune的实现方式

通过**args_grad**来控制是否要进行梯度计算。不论在module init中的fixed_param_names还是通过model中的bind。其实最终都是转化成symbol 的bind。而symbol的bind可以指定**args_grad**用于控制是否需要梯度。



### sgd，momentum理论

不需要把所有数据都加载到内存一起计算。而且在数据有冗余的情况下能够更快的收敛。

SGD:

$$\theta=\theta-\eta\nabla_{\theta}J(\theta;x^{(i)};y^{(i)})$$

SGD的缺点是容易收到噪声的影响。更新不稳定。引入Momentum模拟物体的动量，在一定程度上保持当前更新方向，并且根据当前的batch进行调整，从而确定最终的更新方向。momentum的取值在0-1之间。在训练刚开始时，梯度的变化较大，一般选择较小的动量0.3或者0.5。而到了训练后期，避免的当前批量的噪声数据造成过大的影响，一般选择较大的动量，例如0.9, 与此同时，也会不断的减少学习率，避免震荡。

Momentum: 动量

$$v_t={\gamma}v_{t-1}-\eta\nabla_{\theta}J(\theta)$$

$$\theta=\theta-v_{t}$$

更多优化方法总结，参考[这篇文章](http://sebastianruder.com/optimizing-gradient-descent/index.html),以及[这篇csdn](http://blog.csdn.net/luo123n/article/details/48239963)




