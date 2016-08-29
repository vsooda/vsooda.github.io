---
layout: post
title:  "neuraltalk源码分析"
date:   2015-08-21
categories: ML
tags: rnn
---
* content
{:toc}




### 与论文的异同
没有实现BRNN对齐等等， MultiModal RNN其实就是RNN里面将图像特征作为输入偏置。然后使用单词作为RNN的输入，希望的输出是该单词序列+end

### 总体框架
- driver.py 用于驱动整个训练过程
- predict_on_images.py 用于驱动测试。
- eval_sentence_predictions.py进行评估.
在评估的时候，会讲结果保存到json文件。再在打开该目录下的html文件，即可可视化显示标注结果。

算法主体都在imagernn里面。

- data_provider.py负责转化数据格式便于程序处理。
- generic_batch_generator.py 适配类。用于适配具体的网络模型和损失函数。
- rnn_generator.py 用rnn前向后向更新，并实现束搜索。
- lstm_generator.py与上面的类似，只是使用的是LSTM
- solver.py对更各种优化算法进行封装。求出梯度是固定的，不同的优化算法只是采用不同的参数策略。
- imagernn_util.py, util.py辅助函数。

各种参数含义：

- Ws 是每个词的向量表示
- We 图片的编码
- be 图片表示的偏移量

### 实现
1. ReLU： 一半就是在max（0， exp..)这种

### 流程剖析
该实现基本上各个某块使用静态类+静态方法的方式实现。需要该类的时候就获取一次。像generic_batch_generator.py和rnn_generator.py这种网络结构里面有很多参数。在本实现中，都是在init方法中进行随机初始化，然后把他们放到一个model的json里面进行统一管理。在每个函数内部一半都是先获取一下需要的变量，再进行处理。还将需要优化更新的参数名称放到`update`里面。在更新的时候取出`grads`里面的值，再进行更新。有一些cache结构，是为了保存参数，以便后向算法进行计算

整体执行流程：

- driver.py 读取参数，转换数据，划分训练测试集合
- 初始化batchDecodeGenerator(generic_batch_generator.py), 初始化RNN或者LSTM（rnn_generator.py）
- 循环，迭代到最大次数
- 采样100个句子+图片特征对，构造成batch，作为一次迭代单位
- 调用solver.step(). 传入costfun，供内部调用。这里costfun其实是driver.py的RNNGenCost
- step()内部调用costfun，获取各种梯度，再进行参数更新
- costfun（RNNGenCost）内部， 调用batchDecodeGenerator进行forward，获取预测值。再backward获取梯度值
- 在batchDecodeGenerator把语句，图片都编码为相同维度。对于每个语句，特征对调用rnn_decode.decodeGenerator的forward进行前向计算。
- decodGenerator后向获取梯度
