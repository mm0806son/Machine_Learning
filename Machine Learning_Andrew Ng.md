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

$x(i)$ to denote the “**input**” variables (living area in this example), also called input features, and $y^{(i)}$ to denote the “**output**” or target variable that we are trying to predict (price). A pair $(x^{(i)} , y^{(i)} )$ is called a **training example**, and the dataset that we’ll be using to learn—a list of $m$ training examples$ {(x^{(i)} , y^{(i)} ); i = 1, . . . , m}$—is called a **training set**.

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

## Week 4

> 如果用上面的方法列举所有高次项组合来实现非线性，复杂度会随着参数上升而迅速上升...

### Neural Networks

If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $Θ^{(j)}$ will be of dimension $s_{j+1}×(s_j+1)$.
$$
\begin{array}{r}
a_{1}^{(2)}=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right) \\
a_{2}^{(2)}=g\left(\Theta_{20}^{(1)} x_{0}+\Theta_{21}^{(1)} x_{1}+\Theta_{22}^{(1)} x_{2}+\Theta_{23}^{(1)} x_{3}\right) \\
a_{3}^{(2)}=g\left(\Theta_{30}^{(1)} x_{0}+\Theta_{31}^{(1)} x_{1}+\Theta_{32}^{(1)} x_{2}+\Theta_{33}^{(1)} x_{3}\right) \\
h_{\Theta}(x)=a_{1}^{(3)}=g\left(\Theta_{10}^{(2)} a_{0}^{(2)}+\Theta_{11}^{(2)} a_{1}^{(2)}+\Theta_{12}^{(2)} a_{2}^{(2)}+\Theta_{13}^{(2)} a_{3}^{(2)}\right)
\end{array}
$$
注意要加偏移量$\theta_0$。上标括号里代表的是层数。

<img src="https://raw.githubusercontent.com/mm0806son/Images/main/202109271702585.png" alt="image-20210927170257865" style="zoom: 25%;" />



具体的操作是这样的：
$$
z^{(j)}=\Theta^{(j-1)} a^{(j-1)}
$$

$$
a^{(j)}=g\left(z^{(j)}\right)
$$

直到最后一步：
$$
h_{\Theta}(x)=a^{(j+1)}=g\left(z^{(j+1)}\right)
$$
和 logistic regression 做的是一样的。

> 神经网络可以模拟逻辑电路，也就是说可以用机器算出逻辑电路。

![img](https://raw.githubusercontent.com/mm0806son/Images/main/202109272110134.png)

## Week 5

### **Back propagation Algorithm**

#### Cost Function

> Neural Network的Cost function是Logistic regression的延伸。

For Logistic regression: 
$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}
$$
For Neural Network:
$$
J(\Theta)=-\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K}\left[y_{k}^{(i)} \log \left(\left(h_{\Theta}\left(x^{(i)}\right)\right)_{k}\right)+\left(1-y_{k}^{(i)}\right) \log \left(1-\left(h_{\Theta}\left(x^{(i)}\right)\right)_{k}\right)\right]+\frac{\lambda}{2 m} \sum_{l=1}^{L-1} \sum_{i=1}^{s l} \sum_{j=1}^{s l+1}\left(\Theta_{j, i}^{(l)}\right)^{2}
$$

$K$是层数。

第一部分的求和是对Output Layer的每一层单独算然后求和。具体操作是把矩阵每一项乘起来求和。

```matlab
regularized = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) ); 
J = 1 / m * sum( sum( -class_y.* log(h) -  (1-class_y).*log(1-h) ))+ regularized;
```

The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

之前我们在计算神经网络预测结果的时候我们采用了一种正向传播方法，我们从第一层开始正向一层一层进行计算，直到最后一层的$ℎ\theta(x)$。

我们的目标仍然是找到$\min _{\Theta} J(\Theta)$。现在，为了计算代价函数的偏导数$\frac{\partial}{\partial \Theta_{i, j}^{(l)}} J(\Theta)$，我们需要采用一种**反向传播算法**，也就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层。

#### Neural net gradient function

**Back Propagation 具体的操作方法：**

Given training set $\left\{\left(x^{(1)}, y^{(1)}\right) \cdots\left(x^{(m)}, y^{(m)}\right)\right\}$
- Set $\Delta_{i, j}^{(l)}:=0$ for all $(1, i, j)$, 初始化矩阵

For training example $\mathrm{t}=1$ to $\mathrm{m}$ :

1. $\operatorname{Set} a^{(1)}:=x^{(t)}$

2. Perform forward propagation to compute $a^{(l)}$ for $l=2,3, \ldots, \mathrm{L}$ 
   先正向算出结果

   ![](https://raw.githubusercontent.com/mm0806son/Images/main/202110051630507.png)

3. Using $y^{(t)}$, compute $\delta^{(L)}=a^{(L)}-y^{(t)}$ 算最后一层的误差
   Where $L$ is our total number of layers and $a^{(L)}$ is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in $\mathrm{y}$. 

4. Compute $\delta^{(L-1)}, \delta^{(L-2)}, \ldots, \delta^{(2)}$ using $\delta^{(l)}=\left(\left(\Theta^{(l)}\right)^{T} \delta^{(l+1)}\right) . * a^{(l)} . *\left(1-a^{(l)}\right)$
   The delta values of layer $l$ are calculated by multiplying the delta values in the next layer with the theta matrix of layer $l$. We then element-wise multiply that with a function called $\mathrm{g}^{\prime}$, which is the derivative of the activation function g evaluated with the input values given by $z^{(l)}$.
   The g-prime derivative terms can also be written out as:
   $$
   g^{\prime}\left(z^{(l)}\right)=a^{(l)} \cdot *\left(1-a^{(l)}\right)
   $$

5. $\Delta_{i, j}^{(l)}:=\Delta_{i, j}^{(l)}+a_{j}^{(l)} \delta_{i}^{(l+1)}$ or with vectorization, $\Delta^{(l)}:=\Delta^{(l)}+\delta^{(l+1)}\left(a^{(l)}\right)^{T}$
   Hence we update our new $\Delta$ matrix.

   - $D_{i, j}^{(l)}:=\frac{1}{m}\left(\Delta_{i, j}^{(l)}+\lambda \Theta_{i, j}^{(l)}\right)$, if $j \neq 0 .$
   - $D_{i, j}^{(l)}:=\frac{1}{m} \Delta_{i, j}^{(l)}$ , if $j=0$

$𝑙$ 代表目前所计算的是第几层。
$𝑗$ 代表目前计算层中的激活单元的下标，也将是下一层的第$𝑗$个输入变量的下标。
$𝑖$ 代表下一层中误差单元的下标，是受到权重矩阵中第$𝑖$行影响的下一层中的误差单元
的下标。



FP是正向推导，利用上一步的结果推后面的数值，直到最后得到Output。
BP是逆向推导，利用后一步的误差推前面的误差，直到Layer 2得到Cost Function（Layer 1是原始数据，不需要计算）。

> 其实也可以从前往后使用Automatic Differentiation，但是输出层只有一个单元，所以使用BP。

把$y$先展开到矩阵`class_y`形式，是为了让预测错误时的Loss一样。

### Unrolling parameters

> 把矩阵展开还原成向量用于计算的方法

`thetaVec` 是一个很长的列向量，按顺序把矩阵的所有元素放进去。
`reshape` 是把它还原成矩阵。

```matlab
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]

Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

### Gradient Check

Two-side derivation:
$$
\frac{\partial}{\partial \Theta} J(\Theta) \approx \frac{J(\Theta+\epsilon)-J(\Theta-\epsilon)}{2 \epsilon}
$$
With multiple theta matrices, we can approximate the derivative **with respect to $\Theta_{j}$** as follows:
$$
\frac{\partial}{\partial \Theta_{j}} J(\Theta) \approx \frac{J\left(\Theta_{1}, \ldots, \Theta_{j}+\epsilon, \ldots, \Theta_{n}\right)-J\left(\Theta_{1}, \ldots, \Theta_{j}-\epsilon, \ldots, \Theta_{n}\right)}{2 \epsilon}
$$
代码实现：

```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

检查算出来的导数和BP的基本一致后，要disable gradient check，因为算的很慢。

### Random Initialization

将所有$\theta$权重初始化为零对神经网络不起作用。当我们BP时，所有节点都会重复更新到相同的值。相反，我们可以用以下方法随机地初始化我们的$\Theta$矩阵的权重。

![img](https://raw.githubusercontent.com/mm0806son/Images/main/202110052205184.png)

Hence, we initialize each $\Theta^{(l)}_{ij}$ to a random value between $[-\epsilon,\epsilon]$

代码实现：

```matlab
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

### **Training a Neural Network**

一般来说hidden layer要么只有一层，要么每层数目都相等。

具体流程：

1. Randomly initialize the weights
2. Implement forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

> $J(\Theta)$ 不是凸函数，不一定能找到全局最小值了。

## Week 6

### Debugging

#### Evaluating a Hypothesis (Over & Underfitting)

随机分一部分数据(30%)作为Test set

For linear regression:
$$
J_{\text {test }}(\Theta)=\frac{1}{2 m_{\text {test }}} \sum_{i=1}^{m_{\text {test }}}\left(h_{\Theta}\left(x_{\text {test }}^{(i)}\right)-y_{\text {test }}^{(i)}\right)^{2}
$$
For classification ~ Misclassification error ( 0/1 misclassification error):
$$
\operatorname{err}\left(h_{\Theta}(x), y\right)=\begin{array}{cc}
1 & \text { if } h_{\Theta}(x) \geq 0.5 \text { and } y=0 \text { or } h_{\Theta}(x)<0.5 \text { and } y=1 \\
0 & \text { otherwise }
\end{array}
$$

$$
\text { Test Error }=\frac{1}{m_{\text {test }}} \sum_{i=1}^{m_{t e s t}} \operatorname{err}\left(h_{\Theta}\left(x_{\text {test }}^{(i)}\right), y_{\text {test }}^{(i)}\right)
$$

#### Model Selection

20% Cross Validation set + 20% Test Set

We can now calculate three separate error values for the three different sets using the following method:

1. Optimize the parameters in Θ using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the cross validation set.
3. Estimate the generalization error using the test set with $J_{test}(\Theta^{(d)})$, (d = theta from polynomial with lower error);

用一部分(Train)训练，一部分(CV)用来找最好的算法，再用一部分(Test)去估计预测未知数据的准确性。三部分互不重复，避免使用已经见过的数据影响结果。训练集用于训练不同的模型，验证集用于模型选择。而测试集由于在训练模型和模型选择这两步都没有用到，对于模型来说是未知数据，因此可以用于评估模型的泛化能力。

#### Bias & Variance

Bias = Underfitting -> $J_{train}$ 高 & $J_{cv}$ 高， $J_{CV}(\Theta) \approx J_{train}(\Theta)$
Variance = Overfitting ->  $J_{train} \gg  J_{cv}$ 

<img src="https://raw.githubusercontent.com/mm0806son/Images/main/202110122144499.png" alt="image-20211012214442374" style="zoom: 33%;" />

当 $\lambda$ 较小时，训练集误差较小（过拟合）而交叉验证集误差较大；随着 $\lambda$ 的增加，训练集误差不断增加（欠拟合），而交叉验证集误差则是先减小后增加。

> 注意只有Loss Function 带$\lambda$

<img src="https://raw.githubusercontent.com/mm0806son/Images/main/202110122227323.png" alt="image-20211012222702202" style="zoom:75%;" />

#### Learning Curves

> 注意欠拟合和过拟合是由模型复杂度决定的，不是由样本复杂度决定的...

##### Experiencing high bias (Underfitting)

- **Low training set size**: causes $J_{train}(\Theta)$ to be low and $J_{CV}(\Theta)$ to be high.

- **Large training set size**: causes both $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ to be high with $J_{train}(\Theta)\approx J_{CV}(\Theta)$.

If a learning algorithm is suffering from **high bias**, getting more training data will not **(by itself)** help much.

![](https://raw.githubusercontent.com/mm0806son/Images/main/202110122241170.png)

##### Experiencing high variance  (Overfitting)

- **Low training set size**: $J_{train}(\Theta)$ to be low and $J_{CV}(\Theta)$ to be high. (**Same**)

- **Large training set size**: $J_{train}(\Theta)$ increases with training set size and $J_{CV}(\Theta)$ continues to decrease without leveling off. Also, $J_{train}(\Theta) \lt J_{CV}(\Theta)$ but the difference between them remains significant.

If a learning algorithm is suffering from **high variance**, getting more training data is likely to help.

![](https://raw.githubusercontent.com/mm0806son/Images/main/202110122247590.png)

#### System Design Example

> 一个垃圾邮件分拣器...

我们可以选择一个由100 个最常出现在垃圾邮件中的词所构成的列表，根据这些词是否有在邮件中出现，来获得我们的特征向量（出现为1，不出现为0）。

为了构建这个分类器算法，我们可以做很多事，例如：
1. 收集更多的数据，让我们有更多的垃圾邮件和非垃圾邮件的样本；
2. 基于邮件的路由信息开发一系列复杂的特征；
3. 基于邮件的正文信息开发一系列复杂的特征，包括考虑截词的处理；
4. 为探测刻意的拼写错误（把watch 写成w4tch）开发复杂的算法。

##### Error Analysis

构建一个学习算法的推荐方法为：
1. 从一个简单的能快速实现的算法开始，实现该算法并用交叉验证集数据测试这个算法；
2. 绘制学习曲线，决定是增加更多数据，或者添加更多特征，还是其他选择；
3. 进行误差分析：人工检查交叉验证集中我们算法中产生预测误差的实例，看看这些实例是否有某种系统化的趋势。

以我们的垃圾邮件过滤器为例，误差分析要做的既是检验交叉验证集中我们的算法产生错误预测的所有邮件，看：是否能将这些邮件按照类分组。例如医药品垃圾邮件，仿冒品垃圾邮件或者密码窃取邮件等。然后看分类器对哪一组邮件的预测误差最大，并着手优化。思考怎样能改进分类器。例如，发现是否缺少某些特征，记下这些特征出现的次数。例如记录下错误拼写出现了多少次，异常的邮件路由情况出现了多少次等等，然后从出现次数最多的情况开始着手优化。

> 先实现最基础的算法，再添加东西看看是不是变好了...

##### Precision/Recall

> Skewed Classes：只有0.5%的病人，直接说都没病都比1%正确率的算法好…

<img src="https://raw.githubusercontent.com/mm0806son/Images/main/202110142200784.png" style="zoom:33%;" />
$$
\begin{aligned}
&\text {查准率 Precision }=\frac{\text { True positives }}{\# \text { predicted as positive }}=\frac{\text { True positives }}{\text { True positives }+\text { False positives }} \\
&\text {查全率 Recall }=\frac{\text { True positives }}{\# \text { actual positives }}=\frac{\text { True positives }}{\text { True positives }+\text { False negatives }}
\end{aligned}
$$
把更罕见的数据集定为$y=1$。

提高判断阈值可以提高查准率Precision，降低查全率Recall。反之亦然。

判断算法好坏的依据：
$$
F_{1} \text { Score: } 2 \frac{P R}{P+R}
$$

## Week 7

### Support vector machine (SVM)

支持向量机，数学推导略。

SVM解决的问题是经典的二元分类问题。给出一个分类标准使得样本集可以被最好地分类。在样本特征是二维的情况下，可以用下图表示

<img src="https://pic1.zhimg.com/80/v2-5637ff193cc82ff9fee150ceddd10690_1440w.jpg" alt="img" style="zoom:50%;" />

中间的实线是我们最终需要的分割线，在三维及以上的情况下叫做**划分超平面**。

画圈的样本是距离分割线最近的样本，叫做**支持向量**，这也是支持向量机名字的由来。两条黄色的虚线之间的距离叫做**间隔**，显然，这个间隔越大，也就代表两边的样本离分割线越远，我们得到的分割线就越鲁棒。**只有支持向量（边界上的点）会对结果产生影响。**

注意目标函数的数学表达：
$$
\max _{\boldsymbol{w}, b} \frac{2}{\|\boldsymbol{w}\|}
$$
为了方便计算，将上式取倒数，并对$\|\boldsymbol{w}\|$取平方，得到最终需要优化的目标函数：
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}
$$

### Kernel & Landmarks

给定一个训练示例$x$，我们比较$x$和landmarks的距离来定义它们的相似性：
$$
f_{1}=\operatorname{similarity}\left(x, l^{(1)}\right)=e\left(-\frac{\left\|x-l^{(1)}\right\|^{2}}{2 \sigma^{2}}\right)
$$
这是一个高斯核函数(Gaussian Kernel)。

如果一个训练实例$𝑥$与地标$l$之间的距离近似于0，则新特征$𝑓$近似于$𝑒^{−0} = 1$，如果训练实例$𝑥$与地标$l$之间距离较远，则$𝑓$近似于$𝑒^{−(一个较大的数) }= 0$。

现在我们得到的新特征是**建立在原有特征与训练集中所有其他特征之间距离的基础之上的**.

### Use a SVM

要做feature scaling。

不是所有的kernels都能用，要满足Mercer's Theorem才能确保不发散：(?)

- Linear (nothing)
- Polynomial
- Gaussien
- More esoteric: String kernel, chi-square kernel, histogram intersection kernel （非常罕见）

## Week 8

非监督学习：数据集没有标签。

### K-Cluster算法

```matlab
Repeat {
for i = 1 to m
	c(i) := index (form 1 to K) of cluster centroid closest to x(i)
for k = 1 to K
	μk := average (mean) of points assigned to cluster k
}
```

第一个for 循环是赋值步骤，即：对于每一个样例$𝑖$，计算其应该属于的类。第二个for 循环是聚类中心的移动，即：对于每一个类$𝐾$，重新计算该类的质心。

#### 畸变函数 **Distortion function**

K-均值最小化问题，是要最小化所有的数据点与其所关联的聚类中心点之间的距离之和，因此 K-均值的代价函数（又称畸变函数 **Distortion function**）为：
$$
J\left(c^{(1)}, \ldots, c^{(m)}, \mu_{1}, \ldots, \mu_{K}\right)=\frac{1}{m} \sum_{i=1}^{m}\left\|X^{(i)}-\mu_{c^{(i)}}\right\|^{2}
$$

#### 随机初始化

在运行K-均值算法之前，我们首先要随机初始化所有的聚类中心点：

1. 我们选择𝐾 < 𝑚，即聚类中心点的个数要小于所有训练集实例的数量；
2. 随机选择𝐾个训练实例，然后令𝐾个聚类中心分别与这𝐾个训练实例相等。

K-均值算法有可能会停留在一个局部最小值处，而这取决于初始化的情况。为了解决这个问题，我们通常需要多次运行K-均值算法，每一次都重新进行随机初始化，最后再比较多次运行K-均值的结果，选择代价函数最小的结果。这种方法在𝐾较小的时候还是可行的，但是如果𝐾较大，这么做也可能不会有明显地改善。

#### Elbow Method

![](https://raw.githubusercontent.com/mm0806son/Images/main/202110251829770.png)

### 降维 Dimensionality Reduction

目的：压缩数据 -> 去除冗余度 (Inch and cm) & 方便可视化

### 主成分分析 Principal Component Analysis

找到一个维度更低的超平面。

要先对数据做normalization。

第一步是均值归一化。我们需要计算出所有特征的均值，然后令 $𝑥_𝑗 = 𝑥_𝑗 − 𝜇_𝑗$。如果特征是在不同的数量级上，我们还需要将其除以标准差 $\sigma^2$。
第二步是计算协方差矩阵（covariance matrix）:
$$
\text { Sigma }=\frac{1}{m} \sum_{i=1}^{m}\left(x^{(i)}\right)\left(x^{(i)}\right)^{T}
$$
第三步是计算协方差矩阵𝛴的特征向量（eigenvectors）: `[U, S, V]= svd(sigma)`

如果我们希望将数据从$𝑛$维降至$𝑘$维，我们只需要从$𝑈$中选取前$𝑘$个向量，获得一个$𝑛 × 𝑘$维度的矩阵，我们用$𝑈_{𝑟𝑒𝑑𝑢𝑐𝑒}$表示，然后通过如下计算获得要求的新特征向量：
$$
Z^{(i)}=U_{r e d u c e}^{T} * x^{(i)}
$$
#### Choosing K

我们使用`[U, S, V] = svd(sigma)`得到的S是一个对角矩阵。

假设我们要求误差小于1%：
$$
\begin{gathered}
\frac{\frac{1}{m} \sum_{i=1}^{m}\left\|x^{(i)}-x_{a p p r o x}^{(i)}\right\|^{2}}{\frac{1}{m} \sum_{i=1}^{m}\left\|x^{(i)}\right\|^{2}}=1-\frac{\sum_{i=1}^{k} s_{i i}}{\sum_{i=1}^{n} s_{i i}} \leq 1 \% \\
\frac{\sum_{i=1}^{k} s_{i i}}{\sum_{i=1}^{n} s_{i i}} \geq 0.99
\end{gathered}
$$

### Week 9

#### Anomaly detection

异常检测：

![](https://raw.githubusercontent.com/mm0806son/Images/main/202111021508037.png)



**Anomaly detection algorithm:**

1. Choose features $x_{i}$ that you think might be indicative of anomalous examples.
2. Fit parameters $\mu_{1}, \ldots, \mu_{n}, \sigma_{1}^{2}, \ldots, \sigma_{n}^{2}$
$$
\begin{aligned}
\mu_{j} &=\frac{1}{m} \sum_{i=1}^{m} x_{j}^{(i)} \\
\sigma_{j}^{2} &=\frac{1}{m} \sum_{i=1}^{m}\left(x_{j}^{(i)}-\mu_{j}\right)^{2}
\end{aligned}
$$
3. Given new example $x$, compute $p(x)$ :
   $$
   p(x)=\prod_{j=1}^{n} p\left(x_{j} ; \mu_{j}, \sigma_{j}^{2}\right)=\prod_{j=1}^{n} \frac{1}{\sqrt{2 \pi} \sigma_{j}} \exp \left(-\frac{\left(x_{j}-\mu_{j}\right)^{2}}{2 \sigma_{j}^{2}}\right)
   $$
   Anomaly if $p(x)<\varepsilon$

之前我们构建的异常检测系统也使用了带标记的数据，与监督学习有些相似：

| 异常检测                                                     | 监督学习                                      |
| :----------------------------------------------------------- | --------------------------------------------- |
| 非常少量的正向类（异常数据 𝑦 = 1）, 大量的负向类（𝑦 = 0）    | 同时有大量的正向类和负向类                    |
| 许多不同种类的异常，非常难根据非常少量的正向类数据来训练算法。 | 有足够多的正向类实例，足够用于训练算法。      |
| 未来遇到的异常可能与已掌握的异常、非常的不同。               | 未来遇到的正向类实例可能与训练集中的非常近似. |
| 例如： 欺诈行为检测，生产（例如飞机引擎）检测数据中心的计算机运行状况 | 例如：邮件过滤器，天气预报，肿瘤分类          |

