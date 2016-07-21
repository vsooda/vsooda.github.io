---
layout: post
title: "flappy bird"
date: 2016-07-01
categories: dl
---
###1 mdp

![image](http://vsooda.github.io/assets/flappy_bird/mdp.png)

\\(s\_0,a\_0,r\_1,s\_1,a\_1,r\_2,...,s\_{n-1},a\_{n-1},r\_n,s\_n\\)

针对当前状态\\(s\\)做出行为\\(a\\),获得奖赏值\\(r\\),直到最终达到终止状态\\(n\\)

###2 折扣奖赏
在强化学习的应用场景中，通常存在直到最终状态之前不知道决策好坏的情况，称为奖赏延迟。对于这种情况，上次我们在分析pong的时候使用的是折扣奖赏。就是从最后结果中反推前面每一帧应该获得的奖赏值。

\\( R\_t = \sum\_{k=0}^{\infty} \gamma^t r\_{t+1} \\)

强化学习要做的就是通过某种方式得知当前状态下做出哪种行为将获得更大的奖赏值。训练方法有：Q学习，策略梯度

###3 Q学习

Q学习中定义Q函数：\\(Q(s,a)\\)表示在状态\\(s\\)采取\\(a\\)行为所能获得的最大奖赏值。
$$Q(s,a)=max(R\_{t+1})$$
可以理解Q函数为：在s状态下采取了a行为后，游戏结束所能获得的最高得分。之所以叫做Q函数`Q-function`,因为他代表了在某种状态下采取某种行为的质量`quality`。这是一种值函数近似。

在训练时候我们采集\\(<s,a,r,s'>\\)序列，那么，Q函数可表示为：
$$Q(s,a)=r+{\gamma}max\_{a'}Q(s',a')$$
称为`Bellman equation`

在训练中，最终要的技巧是：`experience replay`。在训练时，收集大量这种四元组序列，再随机采样进行训练。这样避免了大量相似的序列，避免局部最优，而且这样与监督学习更加相似，容易调试。

在Q学习中，通常使用Bellman equation来近似Q函数。简单的Q学习（表格值函数）实现如下：

![image](http://vsooda.github.io/assets/flappy_bird/algo1.png)

###4 deep Q network
deep Q network其实就是使用神经网络来估计Q函数。神经网络的输入时图片，输出是采取各个行为所能获得的Q函数。

![image](http://vsooda.github.io/assets/flappy_bird/network.png)

在flappy bird的例子中，输入时连续四帧图片。输出`READOUT`用来表示在这情况下，是否“跳”所能获得的奖赏。

在学习的时候会对图片做个预处理，就是简单的阈值操作。

![image](http://vsooda.github.io/assets/flappy_bird/preprocess.png)



那么我们要学习的目标就是：

![image](http://vsooda.github.io/assets/flappy_bird/formula.png)

这里\\(Q(s',a')\\)也是神经网络的预测值。我们要优化的目标是当前神经网络对于s状态的估计值于通过后向传递获得的估计值的`残差`。那么，训练时候获得整个运动序列的作用仅仅只是想要知道\\(s\\)状态所能转移的下一个状态\\(s'\\)，也就是当前图片显示的flappy bird在按了一下跳或者不跳后，图片会变成什么样子。

那么这里就涉及到选择策略：探索（exploration）和利用（exploitation）。

### 探索和利用
在s状态下，我们估计出选择跳还是不跳所能获得的奖赏值。跳和不跳会使环境到完全不同的状态\\(s'\\). 这个策略直接决定了我们下面能获得的图片是什么。选择策略主要包括以下两种：

利用：选择当前估计奖赏值最大的行为。这种方法是根据历史经验来选取最优。缺点是可能陷入局部最优。
探索：按照一定概率选择不同的行为。这种方法不会陷入局部最优，缺点是很难收敛。

以上两种方法各有优劣，称为探索-利用窘境。一般需要取折中方案。有一种方法是\\(epsilon\\)贪心探索（ε-greedy exploration）。在每次选择时候，按照\\(epsilon\\)的概率进行随机选择，\\(1-epsilon\\)选择奖赏最大的行为。在实际的操作中，\\(epsilon\\)可能随着时间慢慢减小。
I

### 算法流程

![image](http://vsooda.github.io/assets/flappy_bird/algo2.png)

### 深度Q学习和策略梯度
* 深度Q学习： 用cnn来估计Q函数。要学习的目标是：当前状态的估计Q函数和下一个状态反向获得的奖赏的误差最小。
* 策略梯度：奖赏值是从结束状态往回传递。要学习的目标是：`当前估计的行为的概率和实际采取行为的误差`乘以`反向奖赏值`，













<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>