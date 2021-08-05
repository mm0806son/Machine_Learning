# Introduction of Machine / Deep Learning

By Hung-yi Lee 李宏毅



Machine Learning ≈ Looking for Function

**Different types of Functions Regression:** 

- The function outputs a scalar Ex. 预测明天访问量
- Classification: Given options (classes), the function outputs the correct one. Ex. AlphaGo
- Structured Learning: create something with structure (image, document). Ex. 作画

机器学习的步骤:

1. Function with Unknown Parameters **←** **Domain Knowledge**

2. Define Loss from Training Data

   Loss: how good a set of values is. Loss is a function of parameters

   ***Error Surface***

3. Optimization

   ***Gradient Descent 梯度下降法***

   (Randomly) Pick an initial value $w^{0}$ Compute $\left.\frac{\partial L}{\partial w}\right|_{w=w^{0}}$
   $$
   w^{1} \leftarrow w^{0}-\left.\eta \frac{\partial L}{\partial w}\right|_{w=w^{0}}
   $$
   $\left.\eta \frac{\partial L}{\partial w}\right|_{w=w^{0}} \quad \eta:$ learning rate (自己设定的 ***Hyperparameter***)

   Update $w$ iteratively 

   这种方法只能取到极小值点，不一定能取到最小值。

   *但取不到最小值不是梯度下降法的主要问题(?)*

Model Bias: 来自模型的限制，没有办法模拟真实现状。

> 我们需要一个更复杂的模型……

### All Piecewise Linear Curve

All Piecewise Linear Curve 可以由常数+一系列的蓝色function (Hard Sigmoid)组成，而曲线可以用PLC逼近。

蓝色function用Sigmoid(S型的) Function逼近，恰好满足$\pm \infty$​​时是常数。

$\text { Sigmoid Function }\\$
$$
\begin{aligned}

&y=c \frac{1}{1+e^{-\left(b+w x_{1}\right)}}\\
&=c \operatorname{sigmoid}\left(b+w x_{1}\right)
\end{aligned}
$$
$w$控制斜率，$b$左右移动，$c$控制高度。

<img src="https://raw.githubusercontent.com/mm0806son/Images/main/20210727224120.png" style="zoom: 25%;" />

给每一个Feature分配在不同sigmoid里的权重。Feature指的是前1,2,3...天的播放量。可以写成矩阵形式：
$$
\begin{aligned}
&r_{1}=b_{1}+w_{11} x_{1}+w_{12} x_{2}+w_{13} x_{3} \\
&r_{2}=b_{2}+w_{21} x_{1}+w_{22} x_{2}+w_{23} x_{3} \\
&r_{3}=b_{3}+w_{31} x_{1}+w_{32} x_{2}+w_{33} x_{3} \\
&{\left[\begin{array}{l}
r_{1} \\
r_{2} \\
r_{3}
\end{array}\right]=\left[\begin{array}{l}
b_{1} \\
b_{2} \\
b_{3}
\end{array}\right]+\left[\begin{array}{lll}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23} \\
w_{31} & w_{32} & w_{33}
\end{array}\right]\left[\begin{array}{l}
x_{1} \\
x_{2} \\
x_{3}
\end{array}\right]}
\end{aligned}
$$
整体流程是这样的：

<img src="https://raw.githubusercontent.com/mm0806son/Images/main/20210727225245.png" style="zoom:25%;" />

再把这些东西拉长拼成一个大的向量$\theta$，这样我们就有了一个新的函数$f(\theta)$​，继续做**梯度下降**。

特别的，可以选择资料的一部分作为batch去做梯度下降。

$\boldsymbol{\theta}^{*}=\arg \min _{\boldsymbol{\theta}} L$
- (Randomly) Pick initial values $\boldsymbol{\theta}^{0}$
- Compute gradient $g=\nabla L^{1}\left(\boldsymbol{\theta}^{0}\right)$
**update** $\boldsymbol{\theta}^{1} \leftarrow \boldsymbol{\theta}^{0}-\eta g$
- Compute gradient $\boldsymbol{g}=\nabla L^{2}\left(\boldsymbol{\theta}^{1}\right)$​
**update** $\boldsymbol{\theta}^{2} \leftarrow \boldsymbol{\theta}^{1}-\eta \boldsymbol{g}$​
- Compute gradient $g=\nabla L^{3}\left(\boldsymbol{\theta}^{2}\right)$
**update** $\boldsymbol{\theta}^{3} \leftarrow \boldsymbol{\theta}^{2}-\eta g$

1 **epoch** = see all the batches once

> 模型还可以做更多的变形…

Rectified Linear Unit (ReLU) 斜坡函数 $c\max(0,b+wx_1)$​
两个ReLU叠起来就变成了Hard Sigmoid

ReLU和Sigmoid叫做**Activation function**

> 还可以继续改我们的模型...

把a作为新的参数生产出a'，叫做一个laver。这就是Neuron和Neural Network。
Deep = Many hidden layers

**Overfitting** ：训练过的资料效果好了，没看过的反而变差了。

