---
layout: post
title: "global pooling"
date: 2017-03-09
mathjax: true
categories: ml
tags: cnn
---
* content
{:toc}
去年在微博上看到这篇[文章](http://hacker.duanshishi.com/?p=1733), 文中关于pad值的描述，关于global pooling的奇妙作用让我一直对global pooling年年不忘。 这两天正好看了一下cnn的pad值的valid，same(half), full模式。对cnn本身已经没有疑问，再回来研究一下这个问题。

研究发现这篇博客作者描述有点坑。对于知道border mode，知道global pooling的人来说，完全没有压力，但是对于不熟悉这些名词的人来说，可能会造成误解了。

反思一下造成这个情况的原因，如果不是直接看代码，有看文章的话，不至于这样。如果了解network in network的话，也不至于这样。如果对cnn基础border mode更熟悉的话，同样不至于造成这个问题。所以，即使更新前沿知识是很有必要的，跟踪代码不够，理论上还是要有制高点，否则容易有误解。



### network in network



#### mlpconv

![](/assets/global_pooling/mlpconv.png)

如上图右图所示，使用多层感知机微型网络构成每一层。

为了获得跨通道的特征，使用1x1的卷积核进行卷积。

#### global average pooling

![](/assets/global_pooling/global_avg_pool.jpg)

从图中可以看出，对于最后分为n类的任务，global pooling层有n张feature map。然后每张feature map内部取一个平均值。最后把这n个平均值直接输入到softmax中，得到每一类别的概率。

使用global pooling去掉了全连接层，减少了很多参数。

### 博客文章解析

jjyy了半天，其实是作者不懂border_mode不懂pad值。 通道数最后还是手动计算...`不了解相关只是的话容易被碎碎念带入沟里。`

找到博客作者[原始代码](https://github.com/burness/mxnet-101/blob/master/day5/symbol_inception-resnet-v2.py)与mxnet example集成的代码比较一下就一目了然了。

最后global pooling的作用仅体现在:

```python
_, out_shape,_ = net.infer_shape(data=input_data_shape)
net = mx.symbol.Pooling(net, kernel=out_shape[0][2:4], stride=(2,2), pool_type='avg')
```

变成:

```python
net = mx.symbol.Pooling(net, kernel=(
      1, 1), global_pool=True, stride=(2, 2), pool_type='avg')
```



如果看了这篇引用[论文](https://arxiv.org/pdf/1602.07261.pdf), 知道其实最后是一个global average pooling层，应该不至于造成误会。

### mxnet global pooling实现

``` c++
Assign(out,
       req[pool_enum::kOut],
       scalar<DType>(1.0f / (param_.global_pool ?
                data.shape_[2] * data.shape_[3] :
                param_.kernel[0] * param_.kernel[1])) * \
       pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                     out_shape,
                     param_.global_pool ? data.shape_[2] : param_.kernel[0],
                     param_.global_pool ? data.shape_[3] : param_.kernel[1],
                     param_.global_pool ? 1 : param_.stride[0],
                     param_.global_pool ? 1 : param_.stride[1]));
```

从中可以看出，global pooling仅仅是把pooling核大小改成图像的长宽，使得pooling结果只有一个值...
