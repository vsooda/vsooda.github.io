---
layout: post
title: "end2end tts knowledge"
date: 2018-03-02
mathjax: true
categories: speech
tags: tts
---
* content
{:toc}
最近打算用mxnet复现tacotron[^tacotron]，deepvoice3[^deepvoice3]，wavenet[^wavenet]等论文。后面会写一系列文章记录相关论文细节，以及复现过程中遇到的问题。本文先介绍所需要的预备知识，扫清大家阅读后面文章可能碰到的障碍。

in progress...



文章列表如下:

<img src="http://vsooda.github.io/assets/tts2_knowledge/tts2_papers.png" style="width:300px">


关键知识主要包含:
* seq2seq， encoder，decoder [^seq2seq]
* attention[^attention]
* gated linear unit [^gated_linear]
* conv seq2seq [^convs2s]
* highway network [^highway]
* causal
* softsign


## seq2seq

seq2seq论文[^seq2seq]发表于2014年，目前已经有3000多引用。

seq2seq用来做序列学习非常有效。比如聊天系统，语音识别，ocr，机器翻译。这些应用最显著的特点就是: 输入输出不一样长，没有一一对应关系。现在我们来举个例子:

假设现在有一个翻译任务是想从一个序列(good, morning) 学习到(早, 上, 好)。传统的dnn没办法直接操作。而seq2seq可以。下面开始介绍seq2seq具体细节。同时，我们把输入抽象为: (A, B, C), 输出抽象成(X, Y, Z). 

看一下下面这张图就知道seq2seq是怎么工作的了。

![](http://vsooda.github.io/assets/tts2_knowledge/seq2seq.png)

seq2seq分别encoder和decoder。上图中，左边是encoder，右边是decoder。左边是一个RNN，将输入(A,B,C)输入到网络中，获取固定长度的隐藏向量W。也就是整个(A,B,C)序列被编码成固定长度的向量W。右边是decoder，也是RNN。由W出发，获取输出X，再把X作为输入，获得Y...一直持续下去，直到碰到结束符为止。最后，decoder输出的(X,Y,Z)就是序列预测的结果。

现在我们来思考一个问题。decoder输出的到底是什么？凭什么能对应到任务所需要的所要映射的序列呢？又是怎么知道是否应该结束呢。

* 翻译任务: 输出是具体的词。有个词表，其中包含了结束符。decoder对每个输出做softmax，即可知道到底是哪个词了。但实际解码没有这么简单。考虑到softmax可能有几个词概率差不多的。那么是否应该用这几个词作为输入，继续解码，看看谁的后续概率更高就选谁的。答案是: YES。用这种方式的解码就是我们常说的束搜索。也就是beam search。
* 语音合成任务: 输入是文本，输出是频谱特征。输入的处理方式与其他任务没有区别。输出的频谱特征是一堆连续值，不是分类任务，怎么知道是否结束呢？答案是: 
  * 设置一个最大解码次数，达到了自然停止。最后对合成语音进行端点检测，去掉尾部多余的空音。
  * 设置一个阈值，例如0.2. 如果输出的所有值都这个数字, 就结束。**todo：待确定。碰到句子中间的空音怎么办**

## attention

deepvoice系列，tacotron这些end2end语音合成论文充斥着大量的attention字眼，如果不懂这个，看起来可能会哭。

attention[^attention] 这篇论文提出于2015，目前引用已经将近3000了。相当厉害。attention有很多变种，可以用于seq2seq，也可以不用。最初的论文是用于提升seq2seq的效果，我们就以这个为例。后面的一些扩展大家自己看。

### 动机

seq2seq是将输入通过encoder映射到一个固定向量，然后从这固定向量出发获得解码输出。但这实际上有点违反人类的认知的。比如上面的例子： (good, morning) 翻译到(早，上，好)。“早上“与”morning“有关系，与”good“几乎没有关系。”好“与”good“有关系，与”moring“几乎没有关系。

attention论文认为，简单的将整个输入映射到一个固定向量是不合理的。应该在解码的时候，寻找与其有关系的输入。然后有这些输入一起决定应该输出什么。这句是**attention**

### 细节

要加一些公式了，莫慌，都很简单。

**todo**: 详解

### alignment

怎么知道训练是否有效，一般会将学习到的对齐用图像画出来。比如: 

![](http://vsooda.github.io/assets/tts2_knowledge/alignment.png)

怎么理解呢? 对于翻译任务，输入和输出一般都是呈线性关系的。正常来说，输入序列先出现的词，在输出中，其对应的词也会先出现。上面这张图表示输入输出的响应强度。因为时序性与局部性，一般来说，对齐图上，对角线会比较亮，一般来说，这表明学习基本没有问题。反之，如果这张图很混乱，那么就表明学习是无效的。

ps: 张俊林发表在程序员上的一篇[文章](http://blog.csdn.net/qq_40027052/article/details/78421155)讲解很通俗易懂。上面内容如果看不懂，建议花点时间看看这篇文章。

## gated linear unit

之前的基于神经网络的语言模型一般是用rnn。rnn可以保存长时依赖，所以效果会比较好。但是rnn存在效率问题，而cnn 可以并行。Language Modeling with Gated Convolutional Networks[^gated_linear]这篇文章通过堆叠多层cnn来获取比较的感受野, 同时作者通过gated linear unit来提高性能，最后效果能与rnn相匹敌。

<img src="http://vsooda.github.io/assets/tts2_knowledge/gated.png" style="width:300px">

**todo: 详解**

## conv seq2seq

deepvoice3第一句就是:

> We present Deep Voice 3, a fully-convolutional attention-based neural text- to-speech (TTS) system.

但是明明有全连接层啊。怎么还说自己全卷积。怎么回事？

其实这里的fully-convolutional描述的是seq2seq的方式。

我们正常听到seq2seq，立马就想到用rnn。facebook的这篇论文[^conv_seq2seq]用`fully-convolution`实现了seq2seq。效果不比rnn的差。优势是大大提升了训练速度。花絮是: 谷歌对标conv seq2seq，推出了Attention is all you need[^attention_need]这篇论文，不用rnn，cnn。堆积大量的self attention，用来计算上一层到下一层的映射。这是后话了。

接下来让我们看一下conv seq2seq是怎么实现的。

**todo: 详解**

## highway network

```python
class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)
```

## causa

causal convolution的意思就是，卷积的输出不依赖于未来的输入。这在自回归(autoregressive)模型是必要的。因为只能根据前面的结果预测当前的值。

non-causal convolution就是普通的卷积。

## softsign

deepvoice3这篇论文用到了softsign。

softsign其实就是与tanh类似的一种激活函数。表达式是: 

$$f(x)=\frac{x}{\lvert x\rvert +1}$$

激活图如下: 

<img src="http://vsooda.github.io/assets/tts2_knowledge/softsign.png" style="width:300px">




## 参考文献

[^attention]: Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2015
[^gated_linear]: Yann Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In ICML, 2017.
[^convs2s]: Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann Dauphin. Convolutional sequence to sequence learning. In ICML, 2017.
[^sampleRnn]: Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, and Yoshua Bengio. SampleRNN: An unconditional end-to-end neural audio generation model. In ICLR, 2017
[^wavenet]: Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. WaveNet: A generative model for raw audio. arXiv:1609.03499, 2016
[^attention_need]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv:1706.03762, 2017.
[^highway]: RupeshKumarSrivastava,KlausGreff,andJu ̈rgenSchmidhuber.Highwaynetworks.arXivpreprint arXiv:1505.00387, 2015.
[^seq2seq]: Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pp. 3104–3112, 2014.
[^deepvoice3]: W.Ping,K.Peng,A.Gibiansky,S.O ̈.Arik,A.Kannan, S. Narang, J. Raiman, and J. Miller, “Deep voice 3: 2000- speaker neural text-to-speech,” CoRR, vol. abs/1710.07654, 2017.
[^tacotron]: Y. Wang, R. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss,N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio, Q. Le,Y. Agiomyrgiannakis, R. Clark, and R. A. Saurous, “Tacotron:Towards end-to-end speech synthesis,” in Proceedings of Inter-speech, Aug. 2017.