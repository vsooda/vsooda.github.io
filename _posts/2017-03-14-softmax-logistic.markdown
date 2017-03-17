---
layout: post
title: "彻底理解softmax"
date: 2017-03-14
mathjax: true
categories: ml
tags: ml
---
* content
{:toc}
起因是在用mxnet写wgan的时候, wgan要求loss不去log。而mxnet的[SoftmaxOutput](http://mxnet.io/api/python/symbol.html?highlight=logisticregressionoutput#mxnet.symbol.SoftmaxOutput), [LogisticRegressionOutput](http://mxnet.io/api/python/symbol.html?highlight=logisticregressionoutput#mxnet.symbol.LogisticRegressionOutput) 等都是直接将softmax和loss layer进行整合。

> In the forward pass, the softmax output is returned. In the backward pass, the logit loss, also called cross-entroy loss, is added. 

虽然可以通过MakeLoss来定义自己的损失函数，但是还是挺想知道mxnet内部是怎么做的。



一看源码，有点懵逼，就是只有forward, backward的实现。对于为何如此实现没有只言片语。甚至搜索关键字loss没有任何返回。自己查阅相关文章了。发现说来话长...



本文从最简单的logistic讲起，讲到softmax, softmax loss, 最后讲mxnet SoftmaxOutput的具体实现。



### Logistic

也称为对数几率回归。

#### 形式

一个事件的几率(odds)是指该事件发生和不发生的概率的比值。如果事件发生概率是p, 不发生概率为1-p, 则几率是$\frac{p}{1-p}$.  该事件的对数几率(对数几率函数)是$logit(p)=log(\frac{p}{1-p})$

在对数几率模型中:

$$P(Y=1|x)=\frac{exp(w\cdot x)}{1+exp(w\cdot x)}$$

$$P(Y=0|x)=\frac{1}{1+exp(w\cdot x)}$$

对数几率为:

$$log\frac{P(y=1|x)}{1-p(Y=1|x)}=w\cdot x$$

所以对数几率模型是`输出Y=1的对数几率是由输入ｘ的线性函数表示的模型`。

#### 损失函数

对于给定训练集$T={(x_1,y_1), (x_2,y_2),(x_3,y_3),...,(x_N,y_N)}$, 其中$x_i\in R^n,y_i\in \{0,1\}$. 使用最大似然估计来估计模型参数。

似然函数为:

$$\prod_{i=1}^{N}[P(Y=1|x_i)]^{y_i}[1-P(Y=1|x_i)]^{1-y_i}$$

意即，总的几率等于猜对每个样本的几率的乘积。也就是说，如果一个样本是负样本，则累乘其猜测负样本的概率$P(Y=1\mid x)$，如果一个样本为正样本，则累成其猜测正样本的概率$1-P(Y=1\mid x)$。

求上式的最大值，就是求其－log的最小值。

$$J(w)=-\frac{1}{m}\sum_{i=1}^{N}[y_ilogP(Y=1\mid x_i)+(1-y_i)log(1-logP(Y=1\mid x_i))]$$

上面式子也称为损失函数。



#### 求导

上式子继续约减:

$$J(w)=-\frac{1}{m}\sum_{i=1}^{N}\lbrack y_i log \frac{P(Y=1\mid x_i)}{1-P(Y=1\mid x_i)}+log(1-P(Y=1\mid x_i))\rbrack$$

将$P(Y=1\mid x)=\frac{exp(w\cdot x)}{1+exp(w\cdot x)}$代入上式，得到: 

$$J(w)=-\frac{1}{m}\sum_{i=1}^{N}[y_i(w\cdot x_i)-log(1+exp(w\cdot x_i))]$$

对$w$进行求导:

$$\frac{\partial J(w)}{\partial w}=-\frac{1}{m}\sum_{i=1}^{m}[y_ix_i-\frac{1}{1+exp(w\cdot x_I)}*exp(w\cdot x_i)*x_i]\\=-\frac{1}{m}\sum_{i=1}^{m}[y_ix_i-P(Y=1|x_i)x_i]$$

对于ｗ中的第ｊ个参数，其梯度为:

$$\frac{\partial J(w)}{\partial w_j}=\frac{1}{m}\sum_{i=1}^{m}(P(Y=1|x)-y_i)x_{ij}$$



### softmax

在[cs231n](http://cs231n.github.io/neural-networks-case-study/#grad)中，对softmax可以有一个非常好的认识，但是对具体求导一带而过，只说很简单。。。下面先摘抄cs231n上关于softmax的描述，再进行推导。

#### cs231n

We have a way of evaluating the loss, and now we have to minimize it. We'll do so with gradient descent. That is, we start with random parameters (as shown above), and evaluate the gradient of the loss function with respect to the parameters, so that we know how we should change the parameters to decrease the loss. Lets introduce the intermediate variable \\(p\\), which is a vector of the (normalized) probabilities. The loss for one example is:

$$
p_k = \frac{e^{f_k}}{ \sum_j e^{f_j} } \hspace{1in} L_i =-\log\left(p_{y_i}\right)
$$

We now wish to understand how the computed scores inside \\(f\\) should change to decrease the loss \\(L_i\\) that this example contributes to the full objective. In other words, we want to derive the gradient \\( \partial L_i / \partial f_k \\). The loss \\(L_i\\) is computed from \\(p\\), which in turn depends on \\(f\\). It's a fun exercise to the reader to use the chain rule to derive the gradient, but it turns out to be extremely simple and interpretible in the end, after a lot of things cancel out:

$$
\frac{\partial L_i }{ \partial f_k } = p_k - \mathbb{1}(y_i = k)
$$

Notice how elegant and simple this expression is. Suppose the probabilities we computed were `p = [0.2, 0.3, 0.5]`, and that the correct class was the middle one (with probability 0.3). According to this derivation the gradient on the scores would be `df = [0.2, -0.7, 0.5]`. Recalling what the interpretation of the gradient, we see that this result is highly intuitive: increasing the first or last element of the score vector `f` (the scores of the incorrect classes) leads to an *increased* loss (due to the positive signs +0.2 and +0.5) - and increasing the loss is bad, as expected. However, increasing the score of the correct class has *negative* influence on the loss. The gradient of -0.7 is telling us that increasing the correct class score would lead to a decrease of the loss \\(L_i\\), which makes sense. 

All of this boils down to the following code. Recall that `probs` stores the probabilities of all classes (as rows) for each example. To get the gradient on the scores, which we call `dscores`, we proceed as follows:

```python
dscores = probs
dscores[range(num_examples),y] -= 1
dscores /= num_examples
```

Lastly, we had that `scores = np.dot(X, W) + b`, so armed with the gradient on `scores` (stored in `dscores`), we can now backpropagate into `W` and `b`:

```python
dW = np.dot(X.T, dscores)
db = np.sum(dscores, axis=0, keepdims=True)
dW += reg*W # don't forget the regularization gradient
```

Where we see that we have backpropped through the matrix multiply operation, and also added the contribution from the regularization. Note that the regularization gradient has the very simple form `reg*W` since we used the constant `0.5` for its loss contribution (i.e. \\(\frac{d}{dw} ( \frac{1}{2} \lambda w^2) = \lambda w\\). This is a common convenience trick that simplifies the gradient expression.

#### 推导

下面介绍推导过程。可以参考这篇[文章](http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function)。

softmax概率:

$$p_j = \frac{e^{o_j}}{\sum_k e^{o_k}}$$

损失函数:

$$L = -\sum_j y_j \log p_j$$

softmax层求导:

$$\frac{\partial p_j}{\partial o_i} = \frac{\partial  \frac{e^{o_j}}{\sum_k e^{o_k}}}{\partial o_i}\\=\frac{\frac{\partial e^{o_j}}{\partial e^{o_i}}\sum_k e^{o_k}-e^{o_j}e^{o_i}}{(\sum_k e^{o_k})^2}$$



当$i=j$时, $\frac{\partial e^{o_j}}{\partial e^{o_i}}=e^{o_i}$;

当$i\neq j$时，$\frac{\partial o_j}{\partial o_i}=0$.

所以, 

$$\frac{\partial p_j}{\partial o_i} = p_i(1 - p_i),\quad i = j$$

$$\frac{\partial p_j}{\partial o_i} = -p_i p_j,\quad i \neq j.$$

带入损失函数导数中:



$$\frac{\partial L}{\partial o_i}=-\sum_jy_j\frac{\partial \log p_j}{\partial o_i}=-\sum_jy_j\frac{1}{p_j}\frac{\partial p_j}{\partial o_i}\\=-y_i\frac{1}{p_i}p_i(1-p_i)-\sum_{j\neq i}y_j\frac{1}{p_j}({\color{red}{-p_jp_i}})\\=-y_i(1-p_i)+\sum_{j\neq i}y_j({\color{red}{p_i}})\\=-y_i+\color{blue}{y_ip_i+\sum_{j\neq i}y_j({p_i})}\\=\color{blue}{p_i\left(\sum_jy_j\right)}-y_i=p_i-y_i$$

所以，最终对于softmax层，其反向梯度仅仅是**概率值减去label值**。



### 理解

#### 最大似然

这里我们对上面cs231n进行的简单的解释。

```python
dscores = probs
dscores[range(num_examples),y] -= 1
dscores /= num_examples
dW = np.dot(X.T, dscores)
W += -step_size * dW　#为什么是减？见下文解释
b += -step_size * db
```

上面代码中为什么X.T要乘上dscores. 是什么原理？其实重点在于dscores是损失函数对O的偏导数，而不是一个差值。使用链式法则，推导如下:

$$dscores=\frac{\partial J}{\partial O}$$

$$\frac{\partial O}{\partial W}=X^T$$

所以, 

$$\frac{\partial J}{\partial W}=\frac{\partial J}{\partial O}\cdot \frac{\partial O}{\partial W}=X^T\cdot dscores$$

至于到底应该是$X^T\cdot scores$还是$dscores\cdot X^T$可以由矩阵的维度决定。具体理论参见这篇[文章](http://cs231n.stanford.edu/vecDerivs.pdf)。

#### 加还是减

先考虑一个问题: 为什么不直接去$f'(x)$等于０。这是因为损失函数不是单调函数，可能存在多个局部最优点。



在更新权重的时候为什么是减而不是加？



我们知道在函数$y=f(x)$中，如果$f'(x)>0$表示，ｙ在ｘ处随着ｘ增大而增大。反之则减小。（画图可能更清晰一些）

而我们要求的是损失函数的最小值。也就是说，在$f'(x)>0$时，我们可以通过减小ｘ使得整个loss减小。而在$f'(x)<0$时，我们通过增大x的值来使得整个loss减小。



综上，我们通过将ｘ往负梯度的方向进行调整，从而获得更小的loss值。



### softmax layer

caffe 关于softmax layer的描述。

> Softmax with Loss - computes the multinomial logistic loss of the softmax of its inputs. It’s conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient.

参考文章: [pluskid](http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/), [cs231n](http://cs231n.github.io/linear-classify/#softmax)

#### 单层vs多层

#### 数值稳定性问题



### mxnet实现

关于softmaxoutput前向输入概率，后向计算loss, 并用mshadow求梯度。MakeLoss



### 参考资料

李航: 统计学习方法 p79

周志华: 机器学习  p59

http://deeplearning.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92

http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/

http://cs231n.github.io/linear-classify/#softmax

http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function

http://cs231n.github.io/neural-networks-case-study/#grad

[http://cs231n.stanford.edu/vecDerivs.pdf](http://cs231n.stanford.edu/vecDerivs.pdf)