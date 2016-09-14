---
layout: post
title:  "High-Dimensional Local Binary Patterns for Face Verification"
date:   2015-01-23
categories: cv
tags: lbp landmark
---
* content
{:toc}

HD-LBP用于人脸识别取得接近人类的结果。几乎是目前最好的算法。




主要流程是人脸检测，再对齐。在标定点附近找高维特征，最后根据这个特征可以用cos来计算人脸的相似性。

1. 人脸图片的获得：
opencv人脸检测，2倍大小图片。再检测眼角点将图片旋转为正脸。

2. 高纬度特征的提取方法：
把人脸都放缩到300, 212, 150, 106, 75大小， 再这五个层次分别提取，最后合成高维特征。
对于每个层次的人脸。在每个特征点附近找4x4的cell。每个cell包含10x10像素。根据灰度图邻域像素值关系进行编码，获得LBF特征，再对这些特征进行归一化，使得每种特征最多有两个转移。
每个cell获得59维特征。那么特征总数为：59 x cells x landmarks x 层数
在论文实现中采用16个特征点，4x4cell。那么总的特征维数为：
59 x 4 x 4 x 16 x 5 = 75520
3. 然后使用PCA降维，再计算余弦值获得相似性。（CVPR 13的论文是使用LDA训练分类器。为了降低时间空间复杂度，提出了rotated sparse regression。学习一个稀疏矩阵来替代PCA和linear projection）

实现细节：怎么从10x10像素获得LBP特征？

```
转移：即0，1的变化。01000000转移为2。 10001001转移为4。
2次转移的共有58种序列，再把不是2次转移的作为一个整体。所以讲特征分为59类
```
