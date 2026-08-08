---
title: "线性代数"
order: 1
---
# 线性代数

线性代数是**研究向量空间、矩阵和线性方程组的数学分支**。在 MS for AI 中，我们不讨论有关**空间、方程组**的内容，主要关心线性代数中的**基本运算**，即加、减、乘、除等基本内容。

## 向量与矩阵

### 向量

**“向量是具有大小和方向的量。”** 这是我们在过去听到的最多的一种陈述。比如，我们经常会用**二维向量**来表示速度，**三维向量**来表示空间中的点的位矢。平面中的方向是二维向量，空间中的方向是三维向量，那么**四维向量**是什么？我们中学解决的几何问题很少会是四维的，因此四个数字的向量也找不到什么几何本体。不过大家应该会有一种直觉：**四维向量对应四维空间的方向，而四维空间至少要由四个不共线的四维向量张成。**以此类推，五维向量对应五维空间的方向，五维空间至少要由五个不共线的五维向量张成；六维向量对应六维空间的方向，六维空间至少要由六个不共线的六维向量张成...

那么向量应该可以是任意维的：$n$ 维的向量代表 $n$ 维空间中的方向，$n$ 维空间至少要由 $n$ 个不共线的 $n$ 维向量张成。我们将向量看作是标量的一维有序的序列，是一个标量的“列表“：

$$
\mathbf{x} = \begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix} \in \mathbb{R}^n
$$

在本讲义中，我们一致使用**加粗的小写字体** $\mathbf{x}$ 来表示**向量**，使用 $\mathbf{x}\in\mathbb{R}^n$ 表示 $\mathbf{x}$ 是 $n$ 维向量。并且没有特别声明的情况下，向量为**列向量**。

> [!NOTE] 行向量与列向量
> 可能你暂时不知道向量横着排和竖着排有什么区别。但是这真的是有影响的。
> 
> 行向量：
> $$
> \mathbf{x} = \begin{pmatrix}x_1, x_2, \cdots, x_n\end{pmatrix} \in \mathbb{R}^{1\times n}
> $$
> 使用 $\mathbf{x}\in\mathbb{R}^{1\times n}$ 来区分与列向量的形状不同。而列向量其实也可以写成 $\mathbf{x}\in\mathbb{R}^{n\times 1}$ 。这种形状规定上的差异导致了它们在一些**与形状有关**的计算上不是等价的，比如**矩阵乘法**。

向量的长度与向量的维度是相等的。$n$ 维向量的长度就是 $n$。形状则与向量的**行列表示**有关。$n$ 维列向量的形状为 $(n, 1)$。而 $n$ 维行向量的形状为 $(1, n)$ 。

### 矩阵

向量是标量的一维组织，**矩阵是标量的二维组织**：
$$
\mathbf{X} = \begin{pmatrix}
x_{11} & x_{12} & x_{13} & \cdots & x_{1n} \\
x_{21} & x_{22} & x_{23} & \cdots & x_{2n} \\
x_{31} & x_{32} & x_{33} & \cdots & x_{3n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & x_{m3} & \cdots & x_{mn} 
\end{pmatrix} \in \mathbb{R}^{m\times n}
$$

在本讲义中，我们一致使用**加粗的大写字体** $\mathbf{X}$ 来表示**矩阵**，使用 $\mathbf{X}\in\mathbb{R^{m\times n}}$ 来表示 $m$ 行 $n$ 列的矩阵。使用 $x_{ij}$ 来表示矩阵第 $i$ 行第 $j$ 列的元素。

矩阵中的每一行都可以看作是一个**行向量**，而每一列都可以看作是一个**列向量**。

## 算术

在本部分我们将介绍学习深度学习时会使用到的一些线性代数算术概念。

### 对应位置的算术

对应位置的算术（常被称为 **element-wise 的操作**）指那些**对应位置进行操作，对其它位置没有依赖**的算术。例如对应位置的加减乘除：

$$
\mathbf{z} = \mathbf{x} + \mathbf{y} = \begin{pmatrix}
x_1 + y_1 \\
x_2 + y_2 \\
\vdots \\
x_n + y_n
\end{pmatrix}
$$

这里用加法举例子，而减法、除法同理。但是这里要注意**乘法**不要混淆了。矩阵/向量的乘法在默认语境下**不是**一种对应位置的算术（后面会介绍这一点），因此在本讲义中，我们统一把这种**对应位置的乘法**称之为**逐元素乘法**或 **Hadamard 积**，符号使用 $\odot$：
$$
\mathbf{z} = \mathbf{x} \odot \mathbf{y} = \begin{pmatrix}
x_1y_1 \\
x_2y_2 \\
\vdots \\
x_ny_n
\end{pmatrix}
$$

朴素的**函数**只是对每个位置的元素独立做变换，因此也可以算是一种对应位置的算术：
$$
\mathbf{y} = f(\mathbf{x}) = \begin{pmatrix}
f_1(x_1) \\
f_2(x_2) \\
\vdots \\
f_n(x_n)
\end{pmatrix}
$$
以上举的例子都是向量，但是由于只是对应位置进行操作，**与几何形状没有关系**，因此矩阵和向量是一样的。但是为了方便理解，这里依然写出以上例子的矩阵形式：
$$
\begin{aligned}
\mathbf{Z} &= \mathbf{X} + \mathbf{Y} = \begin{pmatrix}
x_{11} + y_{11} & x_{12} + y_{12} & \cdots & x_{1n} + y_{1n} \\
x_{21} + y_{21} & x_{22} + y_{22} & \cdots & x_{2n} + y_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} + y_{m1} & x_{m2} + y_{m2} & \cdots & x_{mn} + y_{mn}
\end{pmatrix} \\
\\
\mathbf{Z} &= \mathbf{X} \odot \mathbf{Y} = \begin{pmatrix}
x_{11}y_{11} & x_{12}y_{12} & \cdots & x_{1n}y_{1n} \\
x_{21}y_{21} & x_{22}y_{22} & \cdots & x_{2n}y_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1}y_{m1} & x_{m2}y_{m2} & \cdots & x_{mn}y_{mn}
\end{pmatrix} \\
\\
\mathbf{Y} &= f(\mathbf{X}) = \begin{pmatrix}
f_{11}(x_{11}) & f_{12}(x_{12}) & \cdots & f_{1n}(x_{1n}) \\
f_{21}(x_{21}) & f_{22}(x_{22}) & \cdots & f_{2n}(x_{2n}) \\
\vdots & \vdots & \ddots & \vdots \\
f_{m1}(x_{m1}) & f_{m2}(x_{m2}) & \cdots & f_{mn}(x_{mn})
\end{pmatrix}
\end{aligned}
$$

### 转置

**转置（Transpose）**的含义是将矩阵的**行列互换**。具体来说，对于每一个元素，其下标都要行列互换 $x_{ij}\rightarrow x_{ji}$ ：
$$
\mathbf{X}^\top = \begin{pmatrix}
x_{11} & x_{21} & x_{31} & \cdots & x_{m1} \\
x_{12} & x_{22} & x_{32} & \cdots & x_{m2} \\
x_{13} & x_{23} & x_{33} & \cdots & x_{m3} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_{1n} & x_{2n} & x_{3n} & \cdots & x_{mn}
\end{pmatrix}
$$
转置通过在右上角画类似 T 的符号 $\top$ 来表示。对于一个 $m$ 行 $n$ 列的矩阵，其转置后形状会变为** $n$ 行 $m$ 列**。原先的行向量会变成列向量，原先的列向量会变成行向量。

转置可以泛化到向量上。将向量看作**列或行为 1 的特殊矩阵**，那么转置实际上就是向量的**行列转换**：
$$
\mathbf{x}^\top = \begin{pmatrix} x_1, x_2, \cdots, x_n \end{pmatrix}
$$

### 向量内积与外积

向量**内积（inner/dot product）**，我们在中学阶段就已经经常使用了，其定义为两个向量**对应位置相乘并求和**：
$$
\mathbf{x}\cdot\mathbf{y} = \langle\mathbf{x}, \mathbf{y}\rangle = \sum_{i=1}^n x_iy_i
$$

**外积（outer product）**是一种通过两个向量张成一个矩阵的运算：
$$
\mathbf{x}\otimes\mathbf{y} = \begin{pmatrix}
x_1y_1 & x_1y_2 & x_1y_3 & \cdots & x_1y_n \\
x_2y_1 & x_2y_2 & x_2y_3 & \cdots & x_2y_n \\
x_3y_1 & x_3y_2 & x_3y_3 & \cdots & x_3y_n \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_my_1 & x_my_2 & x_my_3 & \cdots & x_my_n
\end{pmatrix}
$$
其中 $\mathbf{x}\in\mathbb{R}^m$、$\mathbf{y}\in\mathbb{R}^n$、$\mathbf{x}\otimes\mathbf{y}\in\mathbb{R}^{m\times n}$。
### 矩阵乘法

**矩阵乘法**是非常重要的算术，它的计算方式与我们过去所使用的标量算术有很大不同。我们将通过一个更加实际的例子来讲解矩阵乘法。

定义两个矩阵 $\mathbf{X}\in\mathbb{R}^{m\times n}$、$\mathbf{Y}\in\mathbb{R}^{n\times k}$。注意，矩阵乘法中，第一个矩阵的列数与第二个矩阵的行数**必须相等**，否则这两个矩阵是**不能进行矩阵乘法**的。而矩阵乘法的结果 $\mathbf{Z}$ 是一个 $m$ 行 $k$ 列的矩阵：

$$
\mathbf{Z} = \mathbf{XY}\in\mathbb{R}^{m\times k}
$$

矩阵 $\mathbf{Z}$ 中的每个元素，通过下式计算得到：
$$
z_{ij} = \sum_{p=1}^n x_{ip}y_{pj} = \mathbf{x}_i^\top \cdot \mathbf{y}_j
$$
即结果矩阵 $\mathbf{Z}$ 中第 $i$ 行第 $j$ 列的值，等于矩阵 $\mathbf{X}$ 第 $i$ 行向量与矩阵 $\mathbf{Y}$ 第 $j$ 列向量的内积。
> [!NOTE]
> 这里 $\mathbf{x}_i$ 需要进行转置的原因是，在我们的语境下 $\mathbf{x}_i$ 是行向量，而向量内积必须是两个形状一样的向量。因此这里将 $\mathbf{x}_i$ 转置为 $\mathbf{x}_i^\top$ 变为列向量，才能与 $\mathbf{y}_j$ 做内积。

以下是矩阵乘法的示意图：

<ThemedImage src="/static/mathmatic/matmul.svg" dark="/static/mathmatic/matmul-dark.svg" alt="矩阵乘法示意" />

图中以 $\mathbf{X}\in\mathbb{R}^{3\times 4}$、$\mathbf{Y}\in\mathbb{R}^{4\times 3}$ 为例，展示了计算 $\mathbf{Z}$ 矩阵第 $1$ 行第 $2$ 列数值的过程。

如果将向量视作行或列为 1 的**特殊矩阵**，那么矩阵乘法还可以用来统一表示向量的内积与外积：
$$
\begin{aligned}
\mathbf{x}\cdot\mathbf{y} &= \mathbf{x}^\top\mathbf{y} \\
\mathbf{x}\otimes\mathbf{y} &= \mathbf{x}\mathbf{y}^\top
\end{aligned}
$$
向量是特殊的矩阵，所以向量也是可以和矩阵做矩阵乘法的，只需要将向量看作是 $1$ 行 $n$ 列的矩阵，或者是 $n$ 行 $1$ 列的矩阵即可（取决于是行向量还是列向量）。
### 逆

首先我们需要先引入**单位矩阵**的概念。单位矩阵指矩阵对角线的值全为 $1$ ，其它位置全为 $0$ 的方阵（方阵就是行数列数相等的矩阵），用符号 $\mathbf{I}$ 或 $\mathbf{E}$ 表示：
$$
\mathbf{I} = \begin{pmatrix}
1 & 0 & 0 & \cdots & 0 \\
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{pmatrix}
$$

本讲义统一使用 $\mathbf{I}$ 来表示单位矩阵。不难发现，单位矩阵与可以进行矩阵乘法的矩阵相乘都是原来的矩阵，它的地位类似标量运算中的 $1$ ：
$$
\begin{aligned}
\mathbf{XI} &= \mathbf{X} \\
\mathbf{IY} &= \mathbf{Y}
\end{aligned}
$$

矩阵的**逆矩阵**定义为，相乘后可以得到单位矩阵 $\mathbf{I}$ 的那个矩阵。通过在右上标注 $-1$ 来表示：
$$
\mathbf{XX}^{-1} = \mathbf{X}^{-1}\mathbf{X} = \mathbf{I}
$$
逆的地位和标量运算中的**倒数**类似。

### 范数

在这里我们介绍三种范数：**$L_1$ 范数、 $L_2$ 范数与 Frobenius 范数**。范数使用 $||$ 包围向量或矩阵来表示，例如 $||\mathbf{x}||_n$ 表示 $\mathbf{x}$ 的 $L_n$ 范数，$||\mathbf{X}||_{\text{Frob}}$ 表示矩阵的 Frobenius 范数。

$L_1$ 范数定义为**向量所有元素的绝对值之和**：
$$
||\mathbf{x}||_1 = \sum_{i=1}^n |x_i|
$$

$L_2$ 范数定义为**向量所有元素的平方和的平方根**：
$$
||\mathbf{x}||_2 = \sqrt{\sum_{i=1}^n x_i^2}
$$

$L_2$ 范数实际上就是向量的长度。

以此类推，$L_p$ 范数定义为：
$$
||\mathbf{x}||_p = (\sum_{i=1}^n x_i^p)^{\frac{1}{p}}
$$

Frobenius 范数定义为**矩阵所有元素平方和的平方根**：
$$
||\mathbf{X}||_{\text{Frob}} = \sqrt{\sum_{ij} x_{ij}^2}
$$

