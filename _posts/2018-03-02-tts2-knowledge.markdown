---
layout: post
title: "tts2 knowledge"
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

通过seq2seq来做聊天系统，做语音识别，ocr，nmt都是非常实用的。seq2seq论文[^seq2seq]发表于2014年，目前已经有3000多引用。

大家对这篇文章应该都比较熟悉吧。这里就简要介绍一下了。不熟悉的同学自行补课 :)

**todo: 详解**

## attention

deepvoice系列，tacotron这些end2end语音合成论文充斥着大量的attention字眼，如果不懂这个，看起来可能会哭。

这里简单介绍一下attention。

**todo: 详解**

张俊林发表在程序员上的一篇[文章](http://blog.csdn.net/qq_40027052/article/details/78421155)讲解很通俗易懂。上面内容如果看不懂，建议花点时间看看这篇文章。

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