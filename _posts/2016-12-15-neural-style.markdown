---
layout: post
title: "neural style"
date: 2016-12-15
categories: ml
tags: deep mxnet
---
* content
{:toc}


本文介绍neural style 进行风格传输。先介绍原始论文，再介绍加速版本基于Perceptual Loss的风格传输。最后介绍该方法在语音传输上的应用。



![](/assets/neural_style/result.png)



### 原始neural style



#### 原理

![](/assets/neural_style/neural_style.png)



通过将用于物体识别的vgg网络不同层进行可视化，可以发现在输入图片之后，低层cnn保存更多的细节（content）。高层cnn保留更多风格（style）相关的内容。neural style的一个直观思路就是尝试结合不同层次的特征，将一张图片的风格传输另一张图片，并保持传输后图片的内容基本保持不变。

下图是更能说明问题：

对于content图片，将vgg各层输出重建图像的结果：

![](/assets/neural_style/content_in_cnn.png)



对于风格图片，各层输出重建结果：

![](/assets/neural_style/style_in_cnn.png)



在实际应用中，content一般采用**relu3\_3**(实际代码中也有采用relu3\_1)特征。style采用**'relu1\_2', 'relu2\_2', 'relu3\_3', 'relu4\_3'**



整体流程：

* 将输入图片（content）扔到网络中，计算content_cnn。
* 将style输入到cnn中得到各层style_cnn, 再计算各层style_cnn的gram矩阵：style_gram
* 将当前迭代的图片current_image输入到cnn（在第一次迭代中，该图片由高斯白噪声随机初始化）。获取content cnn输出记为image_content_cnn, 输入style 中获得image_style_cnn并计算gram矩阵image_gram。 以（content_cnn - image_content_cnn)^2 + （style_gram-image_gram)^2作为loss。反向传播修改vgg各层网络参数，并修改current_image
* 重复以上步骤



#### 算法

![](/assets/neural_style/content_loss.png)





![](/assets/neural_style/style_loss.png)

其中G是style_cnn的gram矩阵。**ps** : gram矩阵的计算见后文。





![](/assets/neural_style/neural_style_loss.png)



上述算法出现之后，艺术化风格吸引了大量的关注。但是由于生成一张图片大约需要一分钟左右时间，所以很难实用。下面介绍另外一个算法，使得艺术化风格可以实时。



###  Perceptual Loss neural style

原始的neural style之所以慢，是因为需要在风格传输的时候，学习网络参数。很多研究者就想，是否可以学习一个网络使得原始图片可以直接转化为目标图片。而这个网络的学习则由另一个网络来判断好坏。这其实是GAN的思想。mxnet的作者之一也是GAN的作者很早就提出这个想法，并实验了该方案，参考这篇[文章](http://dmlc.ml/mxnet/2016/06/20/end-to-end-neural-style.html)。Perceptual Loss neural style思想与其类似。下面介绍Perceptual Loss neural style。



#### 原理

![](/assets/neural_style/perceptual_loss_neural_style.png)



**训练**:

在上图中，输入图片通过一个网络transform net转化为目标图片$\hat{y}$。并将这张图片输入到loss network中，获取feat层输出$y_c$，style层输出$y_s$。这个目标图片输入到loss network中，获得style输出和feat输出。这两个输出与原始图片在网络中的输出对比，构造loss，反向传播优化transform net，loss network保持不变。

使用mscoco数据集，对于每种风格训练一个transform net。并将transform network的参数保存下来。有了这样的训练过程，可以保证对于大多数输入图片的对应风格输出都较好。transform network学习的其实就是neural network loss的最小化参数。

**生成**:

在生成的时候，只要扔到网络中，输出的就是目标图片。不需要优化过程，所以速度很快。



#### 细节

![](/assets/neural_style/feat_loss.png)



![](/assets/neural_style/gram.png)



![](/assets/neural_style/perceptual_style_loss.png)\

**gram 矩阵意义**

在第l层cnn输出$c*h*w$的feature map。 也就是在$h*w$的网格上，每个点都有$c$维特征。gram矩阵（$c*c$),  就是这c维特征的非中心化协方差。

![](/assets/neural_style/style_transfer.png)

### neural style audio

参考[这篇文章](http://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/), [tensorflow代码](https://github.com/DmitryUlyanov/neural-style-audio-tf)。


### 参考

* [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
* [end2end mxnet](http://dmlc.ml/mxnet/2016/06/20/end-to-end-neural-style.html)
* [mxnet perceptual loss](https://github.com/zhaw/neural_style)
* [mxnet neural style](https://github.com/dmlc/mxnet/tree/master/example/neural-style)
* [mxnet zhihu](https://zhuanlan.zhihu.com/p/24205969?refer=gomxnet)
