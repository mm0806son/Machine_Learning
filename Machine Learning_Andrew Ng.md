# Machine Learning 

by Andrew Ng

## Week 1

### Introduction

网页搜索…自动驾驶…AI的未来是模拟人脑的运作方式(神经网络)。

两个定义：

- Arthur Samuel (1959) ：机器学习是gives computers the ability to learn, without being explicitly programmed.

- Tom Mitchell (1998) : "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." 经验E，评价函数P，事件T。

通常来说，所有的机器学习问题都可以归入以下两类：监督学习和非监督学习。Supervised learning and Unsupervised learning.

#### Supervised Learning

给了一个**数据集**作为**正确答案**，我们知道我们要什么：

- Regression: 预测连续的数值（房屋价格）

- Classification: 预测不连续的数值（恶性肿瘤？）

*有更多的参数时就变成高维了？*

#### Unsupervised Learning

给了一个**数据集**，我们不知道我们要什么，能找到某种结构吗？

- Clustering: 给基因分组
- Non-Clustering: 把两个声源的声音分开

### Model and Cost Function

*x*(*i*) to denote the “**input**” variables (living area in this example), also called input features, and $y^{(i)}$ to denote the “**output**” or target variable that we are trying to predict (price). A pair $(x^{(i)} , y^{(i)} )$ is called a **training example**, and the dataset that we’ll be using to learn—a list of $m$ training examples$ {(x^{(i)} , y^{(i)} ); i = 1, . . . , m}$—is called a **training set**.

Our goal is, given a training set, to learn a function $h : X → Y$ so that $h(x)$ is a “good” predictor for the corresponding value of $y$. For historical reasons, this function h is called a **hypothesis**.

**Cost Function**: (Loss) 预测值和真实值的差距。除以2是为了梯度下降时和微分的2约掉。
$$
J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(\hat{y}_{i}-y_{i}\right)^{2}=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x_{i}\right)-y_{i}\right)^{2}
$$
### Parameter Learning

**Gradient descent**: 

> $:=$表示赋值

反复更新$\theta$直至收敛。意义是沿着斜坡走了一小步下山。
$$
\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right)
$$
$$
\begin{aligned}
\frac{\partial}{\partial \theta_{j}} J(\theta) &=\frac{\partial}{\partial \theta_{j}} \frac{1}{2}\left(h_{\theta}(x)-y\right)^{2} \\
&=2 \cdot \frac{1}{2}\left(h_{\theta}(x)-y\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(h_{\theta}(x)-y\right) \\
&=\left(h_{\theta}(x)-y\right) \cdot \frac{\partial}{\partial \theta_{j}}\left(\sum_{i=0}^{n} \theta_{i} x_{i}-y\right) \\
&=\left(h_{\theta}(x)-y\right) x_{j}
\end{aligned}
$$

$$
\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
$$

> 求和部分是矩阵乘法乘出来的标量

$\alpha$ is the **learning rate** (步长). 太小了收敛慢，太大了可能会震荡。

梯度下降法找到的是局部极小值。但是是凸函数，所以极小值就是全局最小值。因为$J$是标准差，是凸函数(convex quadratic function)。

**Batch**: Each step of gradient descent uses all the training examples.

## Week 2

### Multiple features (multivariate linear regression)

$x_i$ 表示第$i$种数据类型，$x^{(j)}$ 表示第$j$组数据。为了记号的方便，定义$x_0^{(i)}=1$。
$$
\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{j}^{(i)} \quad \text { for } \mathrm{j}:=0 \ldots \mathrm{n}
$$

### Gradient descent in PRACTICE (Tips)

- Feature Scaling & Mean Normalization

  通过控制参数范围在0附近来提高收敛速度。
  $$
  x_{i}:=\frac{x_{i}-\mu_{i}}{s_{i}}
  $$
  Where $μ_i$ is the **average** of all the values for feature $(i)$ and $s_i$ is the range of values $(max - min)$, or $s_i$ is the standard deviation.

  没有控制参数范围就会变成这样：

  <img src="https://raw.githubusercontent.com/mm0806son/Images/main/202109141508887.png" style="zoom:25%;" />

- Choose learning rate $\alpha$

  数学证明只要$\alpha$足够小，总会收敛。只不过太小了收敛很慢。

  

  **Debugging gradient descent.** Make a plot with *number of iterations* on the x-axis. Now plot the cost function, $J(\theta)$ over the number of iterations of gradient descent. 如果$J(\theta)$随着迭代变大或者震荡（或者收敛很慢），说明$\alpha$太大了。

  **Automatic convergence test.** Declare convergence if $J(\theta)$ decreases by less than E in one iteration, where $E$ is some small value such as $10^{−3}$. However in practice it's difficult to choose this threshold value.

- Polynomial Regression

  使用其他函数形式。

  *后文会讲到一种方法可以自动找最合适的函数？*

### Computing Parameters Analytically (Normal equation method)

即直接算出来最小值时的$\theta$。

不用做Feature Scaling，不用选$\alpha$，不用迭代。但是$n$大($\ge10000$)的时候算得很慢。

> With the normal equation, computing the inversion has complexity $\mathcal{O}(n^3)$

$$
\theta=\left(X^{T} X\right)^{-1} X^{T} y
$$

> 为什么用这个式子？-> 去查矩阵的微分公式

造成$\left(X^{T} X\right)$不可逆的原因：

- 线性相关的参数 (size in feet$^2$ & in m$^2$)
- 参数量超过了数据量 -> 删掉一部分参数 / use regularization *晚点讲*

但是octave使用pseudo int也能算出正确的解。

## Week 3

> 分类问题用线性回归是很糟糕的，因为很靠右的一个值会把整个直线往右拉，也会给出大于1或小于0的预测值...

### Classification and Representation

Our new form uses the "**Sigmoid Function**", also called the "Logistic Function":
$$
\begin{aligned}
&h_{\theta}(x)=g\left(\theta^{T} x\right) \\
&z=\theta^{T} x \\
&g(z)=\frac{1}{1+e^{-z}}
\end{aligned}
$$
$h_θ(x)$ will give us the **probability** that our output is 1.

The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

### Logistic Regression Model

> 原来的损失函数不是凸函数，不能用极小值当最小值了...

$$
\begin{array}{ll}
J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \operatorname{Cost}\left(h_{\theta}\left(x^{(i)}\right), y^{(i)}\right) & \\
\operatorname{Cost}\left(h_{\theta}(x), y\right)=-\log \left(h_{\theta}(x)\right) & \text { if } \mathrm{y}=1 \\
\operatorname{Cost}\left(h_{\theta}(x), y\right)=-\log \left(1-h_{\theta}(x)\right) & \text { if } \mathrm{y}=0
\end{array}
$$

有如下性质：

$\operatorname{Cost}\left(h_{\theta}(x), y\right)=0$ if $h_{\theta}(x)=y$ 
$\operatorname{Cost}\left(h_{\theta}(x), y\right) \rightarrow \infty$ if $y=0$ and $h_{\theta}(x) \rightarrow 1$
$\operatorname{Cost}\left(h_{\theta}(x), y\right) \rightarrow \infty$ if $y=1$ and $h_{\theta}(x) \rightarrow 0$

可以化简为：
$$
\operatorname{Cost}\left(h_{\theta}(x), y\right)=-y \log \left(h_{\theta}(x)\right)-(1-y) \log \left(1-h_{\theta}(x)\right)
$$

$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
$$

A vectorized implementation is:
$$
\begin{aligned}
&h=g(X \theta) \\
&J(\theta)=\frac{1}{m} \cdot\left(-y^{T} \log (h)-(1-y)^{T} \log (1-h)\right)
\end{aligned}
$$
Gradient：
$$
\begin{aligned}
&\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)\\
&\theta_{j}:=\theta_{j}-\frac{\alpha}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}\\
&\theta:=\theta-\frac{\alpha}{m} X^{T}(g(X \theta)-\vec{y})
\end{aligned}
$$
和之前线性回归方法的公式是一样的，但是这里的$h(x)$变了。

还有很多优化方法：Conjugate gradient, BFGS, L-BFGS.. 不需要手动选择$\alpha$，通常比梯度下降要快。但是更复杂。

Octave里已经包含了一些函数：

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);

```

```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

### Multiclass Classification

思路是分别区分每一个类型：

<img src="https://raw.githubusercontent.com/mm0806son/Images/main/202109231136281.png" style="zoom: 25%;" />

### Solving the Problem of Overfitting

Underfit -> High bias

Overfit -> High **variance**

There are two main options to address the issue of overfitting:

1) Reduce the number of features:

- Manually select which features to keep.
- Use a model selection algorithm (studied later in the course).

2) Regularization

- Keep all the features, but reduce the magnitude of parameters $\theta_j$.
- Regularization works well when we have a lot of slightly useful features.

越简单的Hypothesis越不容易Overfit。

#### Regularization in Linear Regression

我们在罚函数中添加第二项对系数$\theta$进行控制（除了$\theta_0$以外）： 
$$
\min _{\theta} \frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n} \theta_{j}^{2}
$$
公式变为：
$$
\begin{aligned}
&\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)} \\
&\theta_{j}:=\theta_{j}-\alpha\left[\left(\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}\right)+\frac{\lambda}{m} \theta_{j}\right] \quad j \in\{1,2 \ldots n\}\\

\end{aligned}
$$
即：
$$
\theta_{j}:=\theta_{j}\left(1-\alpha \frac{\lambda}{m}\right)-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
$$
$1−α\frac{λ}{m}$总是比1小一些，因此每次迭代都会把$\theta_j$缩小一点。第二项则和regularization之前的形式一样。

#### **Normal Equation**

> 同样可以直接算出来

$$
\begin{aligned}
&\theta=\left(X^{T} X+\lambda \cdot L\right)^{-1} X^{T} y\\
&where\ L=\left[\begin{array}{lllll}0 & & & & \\ & 1 & & & \\ & & 1 & & \\ & & & \ddots & \\ & & & & 1\end{array}\right]
\end{aligned}
$$
$X^{T} X+\lambda \cdot L$ 永远可逆，解决了Week2提到的可能不可逆的问题。


$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}
$$
The second sum, $\sum_{j=1}^n \theta_j^2$ **means to explicitly exclude** the bias term, $\theta_0$. I.e. the $\theta$ vector is indexed from 0 to n (holding n+1 values, $\theta_0$ through $\theta_n$). Thus, when computing the equation, we should continuously update the two following equations:

![](https://raw.githubusercontent.com/mm0806son/Images/main/202109231603063.png)
