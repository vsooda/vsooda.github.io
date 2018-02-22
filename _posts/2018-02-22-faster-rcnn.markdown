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
本文介绍faster rcnn系列。很久以前看别人博客，整个看下来就是罗列一些已知结论，看完之后还是云里雾里。本文尽量避免这种情况。先正向介绍大体发展流程。再介绍一吨的细节。



先简略看一下整个发展流程。第一遍看不清楚没关系，后面可以回来再看。

## 发展过程

### rcnn

rcnn[^rcnn]是这个系列的开端。rcnn是regions with cnn features的缩写。基本思路如下图所示。用select search等算法对每张图片提取2000个候选区域。然后将这些候选区域扔到cnn网络去，提取特征，使用svm进行分类。记得当时训练的时候，需要100多g硬盘来保存特征。现在看来是多么不可思议。但当时这个网络比传统的算法在voc2012上，将map提高了30%，达到了53.3%.（当然，用现在的眼光来看是很低的)

![](../assets/faster_rcnn/rcnn.png)

训练流程:

* 在imagenet训练cnn
* 训练svm。之所以用svm而不是直接用softmax，是因为作者实验发现svm效果更好
* 训练bbox regression error。对框的位置进行微调

下面对bbox regression进行详细说明。



微调目标:

$$t_x=(G_x-P_x)/P_w$$

$$t_y=(G_y-p_y)/P_h$$

$$t_w=log(G_w/P_w)$$

$$t_y=log(G_h/P_h)$$



### sppnet，fast rcnn

![](../assets/faster_rcnn/fast.png)



#### sppnet

![](../assets/faster_rcnn/spp.png)





![](../assets/faster_rcnn/spp_framework.png)



#### fast rcnn



### faster rcnn

![](../assets/faster_rcnn/anchors.png)

#### rpn



好了上面介绍了整个发展过程。接下来，让我们抛开这些条条框框。让我们直接看rcnn的集大成者faster rcnn里面到底蕴含着哪些细节。首先我们先付下一下faster rcnn是什么？

## 细节

> 刚开始有点误会。以为是anchor是在fast rcnn里面。很矛盾。后面发现anchors是在rpn里。然后看起来就没有疑问了。

### 定义



> Region Proposal Network solves object detection as a regression problem from the objectness perspective. Bounding boxes are predicted by applying learned bounding box deltas to base boxes, namely anchor boxes across different positions in feature maps. Training process directly learns a mapping from raw image intensities to bounding box transformation targets.
>
> Fast R-CNN treats general object detection as a classification problem and bounding box prediction as a regression problem. Classifying cropped region feature maps and predicting bounding box displacements together yields detection results. Cropping feature maps instead of image input accelerates computation utilizing shared convolution maps. Bounding box displacements are simultaneously learned in the training process.
>
> Faster R-CNN utilize an alternate optimization training process between RPN and Fast R-CNN. Fast R-CNN weights are used to initiate RPN for training. The approximate joint training scheme does not backpropagate rcnn training error to rpn training.



### rpn细节

anchors what and why？

### fast rcnn细节

**偏移**：

![](../assets/faster_rcnn/fast_rcnn.png)


**smooth l1损失**:

$$
smooth_{L1}(x) =
\begin{cases}
0.5x^2  & \text{if $\mid x\mid < 1 $ } \\
\mid x\mid - 0.5 & \text{otherwise.}
\end{cases}
$$

**roi映射**: 

fast rcnn的映射沿用sppnet的计算方式。但sppnet也没有很详细的描述这个问题。这篇[博客](https://zhuanlan.zhihu.com/p/24780433), 对其进行较详细的解析。**todo: 看源码后再来补充细节**.

### 训练细节

**todo:** rpn，anchors，fast rcnn的坐标空间分别是什么？0-1？什么时候需要进行映射？在计算proposal的时候映射到整数空间？fast rcnn的偏移也是整数（当然，不一定需要整数，知道原图的大小，proposal大小也就可以用小数？）？

faster rcnn训练需要resize吗？

对于无理取闹的proposal需要训练吗？直接丢弃？: rcnn append c描述这个问题。



roi pooling里，一个激活值可能对多个pooling结果有影响。在反向传播的时候需要对多个累加多个反向梯度。



## 参考文献

[^sppnet]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, & Jian Sun. “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition”. ECCV 2014.
[^fast_rcnn]: R. Girshick, “Fast R-CNN,” in IEEE International Conference on Computer Vision (ICCV), 2015.
[^faster_rcnn]: S.Ren,K.He,R.Girshick,and J.Sun,“FasterR-CNN:Towards real-time object detection with region proposal networks,” in Neural Information Processing Systems (NIPS), 2015.
[^rcnn]: R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic seg- mentation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014s