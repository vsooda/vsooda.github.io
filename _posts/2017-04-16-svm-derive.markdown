---
layout: post
title: "svm derive"
date: 2017-04-16
mathjax: true
categories: ml
tags: svm
---
* content
{:toc}

svm推导，详见pluskid博客。

推导如下，代码待实现。



#### 基本推导

![](/assets/kernels/svm_derive.jpg)



#### smo优化



![](/assets/kernels/smo_line.png)



$\alpha_1,\alpha_2$的取值在如图所示的直线上。假定$\alpha_2$的上限下限分别是$H,L$

![](/assets/kernels/smo_bound.png)

当$y_1 \neq y_2$时，$\alpha_1-\alpha_2=k$

$\alpha_1 - \alpha_2 = C-H$ 所以: $H=C-\alpha_1+\alpha_2$

$\alpha_1 - \alpha_2=0-L$ 所以: $L=\alpha_2 - \alpha_1$

同理,

当$y_1==y_2$时, $\alpha_1+\alpha_2=k$

$\alpha_1+\alpha_2=０+H$, 所以，$H=\alpha_1+\alpha_2$

$\alpha_1+\alpha_2=C+L$, 所以，$L=\alpha_1+\alpha_2-C$



$$\alpha_2^{new}=\alpha_2 + \frac{y_2(E_1-E_2)}{\eta}$$

其中

$$\eta=k(x_1,x_2)+k(x_2,x_2)-2k(x_1,x_2)$$

$$E_i=predict_i-y_i$$

因为，$\alpha_2$必须在$L,H$的范围内，所以如果超出范围需要进行截断clip.
$$
\alpha_2^{new,clip} =
\begin{cases}
H  & \text{if $\alpha_2^{new}\geq H$} \\
\alpha_2^{new} & \text{if $L<\alpha_2^{new}<H$} \\
L & \text{$\alpha_2^{new}\leq L$}
\end{cases}
$$
$\alpha_1$与$\alpha_2$是线性关系，所以，

$$\alpha_1^{new}-\alpha_1=y_1 y_2(\alpha_2^{new,clip}-\alpha_2)$$

所以,

$$\alpha_1^{new}=\alpha_1+y_1y_2(\alpha_2^{new,clip}-\alpha_2)$$

**求ｂ**:其实就是希望b的取值能够使得当前的预测值与标签相同。

$$f(x)=\sum_{j=1}^{N}y_j\alpha_jK(x_j,x)-b$$

记$y_j\alpha_1K(x_j,x)=s_j$

所以，$f(x_1)=s_1+s_2+constant-b$

$f(x_1)^{new}=s_1^{new}+s_2^{new}+constant-b^{new}$

求差值: $f(x_1)^{new}-f(x_1)=s_1^{new}-s_1+s_2^{new}-s_2-(b^{new}-b)$

我们希望这个这个差值能够弥补之前的预测误差，即，

$E_1+f(x_1)^{new}-f(x_1)=E_1+ s_1^{new}-s_1+s_2^{new}-s_2-(b^{new}-b)=0$

从上式子可以求出$b$的一个取值$b_1$。

同理，对于样本$x_2$同样可以获得一个取值$b_2$.

最后采用策略选取$b_1,b_2$.

* 如果a1在边界上，则取b1
* 否则，如果a2在边界上，则取b2
* 否则，取b1,b2的均值。



#### 代码

来自[MLAlgorithm](https://github.com/rushter/MLAlgorithms)

```python
# coding:utf-8
from mla.base import BaseEstimator
from mla.svm.kernerls import Linear
import numpy as np
import logging

np.random.seed(9999)

"""
References:
The Simplified SMO Algorithm http://cs229.stanford.edu/materials/smo.pdf
"""


class SVM(BaseEstimator):
    def __init__(self, C=1.0, kernel=None, tol=1e-3, max_iter=100):
        """Support vector machines implementation using simplified SMO optimization.

        Parameters
        ----------
        C : float, default 1.0
        kernel : Kernel object
        tol : float , default 1e-3
        max_iter : int, default 100
        """
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        if kernel is None:
            self.kernel = Linear()
        else:
            self.kernel = kernel

        self.b = 0
        self.alpha = None
        self.K = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])
        self.alpha = np.zeros(self.n_samples)
        self.sv_idx = np.arange(0, self.n_samples)
        return self._train()

    def _train(self):
        iters = 0
        while iters < self.max_iter:
            iters += 1
            alpha_prev = np.copy(self.alpha)

            for j in range(self.n_samples):
                # Pick random i
                i = self.random_index(j)

                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                if eta >= 0:
                    continue
                L, H = self._find_bounds(i, j)

                # Error for current examples
                e_i, e_j = self._error(i), self._error(j)

                # Save old alphas
                alpha_io, alpha_jo = self.alpha[i], self.alpha[j]

                # Update alpha
                self.alpha[j] -= (self.y[j] * (e_i - e_j)) / eta
                self.alpha[j] = self.clip(self.alpha[j], H, L)

                self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_jo - self.alpha[j])

                # Find intercept
                b1 = self.b - e_i - self.y[i] * (self.alpha[i] - alpha_jo) * self.K[i, i] - \
                     self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j]
                b2 = self.b - e_j - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j] - \
                     self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j]
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)

            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break
        logging.info('Convergence has reached after %s.' % iters)

        # Save support vectors index
        self.sv_idx = np.where(self.alpha > 0)[0]

    def _predict(self, X=None):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self._predict_row(X[i, :]))
        return result

    def _predict_row(self, X):
        k_v = self.kernel(self.X[self.sv_idx], X)
        return np.dot((self.alpha[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T) + self.b

    def clip(self, alpha, H, L):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def _error(self, i):
        """Error for single example."""
        return self._predict_row(self.X[i]) - self.y[i]

    def _find_bounds(self, i, j):
        """Find L and H such that L <= alpha <= H.
        Also, alpha must satisfy the constraint 0 <= αlpha <= C.
        """
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H

    def random_index(self, z):
        i = z
        while i == z:
            i = np.random.randint(0, self.n_samples - 1)
        return i
```





参考: 



http://blog.pluskid.org/?page_id=683

http://cs229.stanford.edu/notes/cs229-notes3.pdf

http://cs229.stanford.edu/materials/smo.pdf

https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf