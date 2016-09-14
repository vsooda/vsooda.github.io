---
layout: post
title: "prosody conversion from neutral speech to emotional speech"
date: 2016-05-24
categories: speech
tags: speech prosody
---
* content
{:toc}

音色转化


###  数据准备
* 准备1500句子，每种句子用五种情绪读出：neutral，happiness，sadness，fear，anger
* 人工对每个句子的情感强度进行标定。分为：strong，medium，weak，unlike
* 重音标定。3个人对每个句子进行标定。3个人都认为重音，则标为3；2个人认为是1则认为是2,依此类推。

原文（可忽略）：

![image](http://vsooda.github.io/assets/prosody/data_record.png)

### 特征
`对现有特征可改进之处`：

* 对于距离phrase,word多少个syllable采用阶段函数。例如，在边界上标为0，离边界1则标为1，2个或者两个以上则认为2.
* word在marytts是token。但是marytts只是把一个断句当做phrase。是否一个短语或者一个连续发音序列就是一个phrase

### f0异同

同一句话在不同情绪，不同情绪强度上，f0有较大的偏差。`情绪需要标注`

总的来说，在happiness,anger时f0较高，在fear，sadness时候f0偏低。

![image](http://vsooda.github.io/assets/prosody/prosody_f0.png)

### f0转化
不同情绪的f0长度也不相同，但是可以使用一个简单的scale处理。f0模型的转化提到三种算法，包括：LMM（linear modification model），GMM，CART

文章同时表明，仅仅是改变f0还是不够，也需要对频率信息进行更改。

>  Tao J, Kang Y, Li A. Prosody conversion from neutral speech to emotional speech[J]. Audio, Speech, and Language Processing, IEEE Transactions on, 2006, 14(4): 1145-1154.
