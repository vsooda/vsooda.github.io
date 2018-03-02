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
最近打算用mxnet复现tacotron[^tacotron]，deepvoice3[^deepvoice3]，wavenet[^wavenet]等论文。后面会写一系列文章记录相关论文细节，以及复现过程中遇到的问题。本文先介绍所需要的预备知识，扫清可能碰到的障碍。



文章列表如下:

<img src="/assets/tts2_knowledge/tts2_papers.png" style="width:300px">


关键知识主要包含:
* seq2seq， encoder，decoder [^seq2seq]
* attention[^attention]
* gated linear unitn [^gated_linear]
* conv seq2seq [^convs2s]
* highway network [^highway]
* causal
* softsign



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