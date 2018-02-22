---
layout: post
title: "faster rcnn系列"
date: 2018-02-22
mathjax: true
categories: deep
tags: cv object
---
* content
{:toc}
本文介绍faster rcnn系列。很久以前看别人博客，整个看下来就是罗列一些已知结论，看完之后还是云里雾里。本文尽量避免这种情况。先正向介绍大体发展流程。再介绍**一吨**的细节。





先简略看一下整个发展流程。第一遍看不清楚没关系，后面可以回来再看。

## 发展过程

### rcnn

rcnn[^rcnn]是这个系列的开端。rcnn是regions with cnn features的缩写。基本思路如下图所示。用select search等算法对每张图片提取2000个候选区域。然后将这些候选区域扔到cnn网络去，提取特征，使用svm进行分类。记得当时训练的时候，需要100多g硬盘来保存特征。现在看来是多么不可思议。但当时这个网络比传统的算法在voc2012上，将map提高了30%，达到了53.3%.（当然，用现在的眼光来看是很低的)

![](http://vsooda.github.io/assets/faster_rcnn/rcnn.png)

训练流程:

* 在imagenet训练cnn。再裁剪物体roi使用softmax分类训练。
* 训练svm。裁剪proposals位置的图片，resize到固定大小，再输入cnn获取特征，然后拿这个特征来训练svm。之所以用svm而不是直接用softmax，是因为作者实验发现svm效果更好
* 训练bbox regression error。对框的位置进行微调

下面对bbox regression进行详细说明。

#### bbox regression

对于proposal P，找到一个最近的标注物体ground-true G。如果两者的IoU大于一个阈值(0.6), 则进行训练。否则，抛弃这个proposal，因为训练即使拿来训练也没有什么意义。

bbox regression的目标就是将弥补proposal和ground-true的偏移。

微调目标:

$$t_x=(G_x-P_x)/P_w$$

$$t_y=(G_y-p_y)/P_h$$

$$t_w=log(G_w/P_w)$$

$$t_y=log(G_h/P_h)$$

### sppnet，fast rcnn

rcnn计算非常慢。主要是因为对每个proposal都重新计算卷积特征。如下图左图所示。那么一个很自然的想法就是，是否可以重复利用这些特征呢？答案是可以。如下图右图所示。sppnet[^sppnet]，fast rcnn[^fast_rcnn]先统一计算feature map。然后将proposal映射到feature map上。然后拿这些roi位置的feature map来进行后续计算。

<img src="http://vsooda.github.io/assets/faster_rcnn/fast.png" style="width:500px">

下面我们分别看一下sppnet，fast rcnn的区别。

#### sppnet

<img src="http://vsooda.github.io/assets/faster_rcnn/spp.png" style="width:500px">

上图表明了sppnet的基本思路。对于一个物体的roi位置的特征图。进行几个尺度的pooling。然后拼接成固定的长度。再通过前连接层获取特征。结构如下:

<img src="http://vsooda.github.io/assets/faster_rcnn/spp_framework.png" style="width:500px">

sppnet的将任意的roi feature map划分到固定网格，再进行pooling的操作被称为: **RoiPooling**.

sppnet的相对于rcnn有个缺点，RoiPooling后面的网络层无法被微调，因为只有检测任务有这些层，分类任务没有（分类并没有roi这种东西）。

#### fast rcnn

* 更高的map
* sppnet和rcnn使用一样的多阶段训练。fast rcnn将svm改成softmax，并使用multitask loss同时进行分类和bbox regrresion。所以只需要一阶段训练就可以了，
* 不需要额外的硬盘存储空间。
* 可以更新所有层 （上面有提到sppnet为什么不能更新所有层）

fast rcnn还采用smooth L1损失函数。相对于L2更加鲁棒。无须精细的调节学习率以避免梯度爆炸。

**smooth l1损失**:

$$
smooth_{L1}(x) =
\begin{cases}
0.5x^2  & \text{if $\lvert x\rvert < 1 $ } \\
\lvert x\rvert - 0.5 & \text{otherwise.}
\end{cases}
$$

### faster rcnn

我们先总结一下fast rcnn还有什么问题。fast rcnn用select search等算法来先获得候选框，再拿这些候选框来分类以及回归出偏移。在实用中，select search一般是cpu代码，限制了整体的效率。也限制了性能的提升。

faster rcnn[^faster_rcnn]提出rpn网络来进行候选框提取。rpn也是神经网络。与fast rcnn网络共享底层，减少大量计算。使得最终达到5fps左右。

<img src="http://vsooda.github.io/assets/faster_rcnn/rpn_share.png" style="width:500px">

从上图我们可以看出，图片经过一系列的卷积层之后，获得feature maps。一方面，拿这些feature map扔到rpn网络中，预测出可能的候选区域proposals。再将这些候选区域的feature maps roi扔到fast rcnn中进一步分类和偏移回归。

接下来我们看一下rpn的具体构造.

#### rpn

<img src="http://vsooda.github.io/assets/faster_rcnn/anchors.png" style="width:500px">

上图是rpn的基本结构。

* 对feature map进行sliding window操作。等价于直接用3x3的卷积核
* anchors。使用anchors来对位置进行编码。从而在一个滑动窗口内部可以获得多个尺度的候选框。rpn学习的相对于anchors box的delta。

刚看的时候有点误会。以为是anchor是在fast rcnn里面。很矛盾。后面发现anchors是在rpn里。然后看起来就没有疑问了。



## 细节

好了上面介绍了整个发展过程。接下来，让我们抛开这些条条框框。让我们直接看rcnn的集大成者faster rcnn里面到底蕴含着哪些细节。首先我们先付下一下faster rcnn是什么？

### 定义

faster rcnn由rpn+fast rcnn构成。rpn和fast rcnn共享基础网络，可以大大提升检测效率。rpn预测出候选框位置，再截取候选框位置的feature map放入fast rcnn进行分类和偏移计算。训练的时候可以采取两者轮流计算的方式，也可以采用（近似）联合训练。

rpn引入了anchors box。fast rcnn需要处理roi 映射。

下面让我们一一解答一下问题: 

* anchors assign问题。坐标空间是什么。
* anchor box的作用是什么
* 损失函数？
* faster rcnn训练需要resize吗？
* 对于无理取闹的proposal需要训练吗？直接丢弃？: rcnn append c描述这个问题。
* roi如何映射？
* **todo:** rpn，anchors，fast rcnn的坐标空间分别是什么？0-1？什么时候需要进行映射？在计算proposal的时候映射到整数空间？fast rcnn的偏移也是整数?（当然，不一定需要整数，知道原图的大小，proposal大小也就可以用小数？）？



### rpn细节

anchors what and why？

**anchor box**：

$$t_x=(x-x_a)/w_a, t_y=(y-t_a)/h_a$$

$$t_w=log(w/w_a), t_h=log(h/h_a)$$

$$t_x^*=(x^*-x_a)/w_a, t_y^*=(y^*-t_a)/h_a$$

$$t_w^*=log(w^*/w_a), t_h^*=log(h^*/h_a)$$

其中$x$, $x_a$, $x^*$分别代表预测box，anchor box， ground-true box。

### fast rcnn细节

**偏移**：

<img src="http://vsooda.github.io/assets/faster_rcnn/fast_rcnn.png" style="width:500px">




**roi映射**: 

fast rcnn的映射沿用sppnet的计算方式。但sppnet也没有很详细的描述这个问题。这篇[博客](https://zhuanlan.zhihu.com/p/24780433), 对其进行较详细的解析。**todo: 看源码后再来补充细节**.

### 训练细节

roi pooling里，一个激活值可能对多个pooling结果有影响。在反向传播的时候需要对多个累加多个反向梯度。



## 参考文献

[^sppnet]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, & Jian Sun. “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition”. ECCV 2014.
[^fast_rcnn]: R. Girshick, “Fast R-CNN,” in IEEE International Conference on Computer Vision (ICCV), 2015.
[^faster_rcnn]: S.Ren,K.He,R.Girshick,and J.Sun,“FasterR-CNN:Towards real-time object detection with region proposal networks,” in Neural Information Processing Systems (NIPS), 2015.
[^rcnn]: R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic seg- mentation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014s