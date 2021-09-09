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

**Cost Function**: (Loss) 预测值和真实值的差距。除以2是为了梯度下降时方便。
$$
J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(\hat{y}_{i}-y_{i}\right)^{2}=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x_{i}\right)-y_{i}\right)^{2}
$$
**Gradient descent**: 

> $:=$表示赋值

反复更新$\theta$直至收敛。意义是沿着斜坡走了一小步下山。
$$
\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right)
$$
$\alpha$ is the **learning rate**. 太小了收敛慢，太大了可能会震荡。

梯度下降法找到的是局部极小值。但是是凸函数，所以极小值就是全局最小值。因为$J$是标准差，是凸函数(convex quadratic function)。

**Batch**: Each step of gradient descent uses all the training examples.

*Normal equation method? 不用迭代*

### Parameter Learning

