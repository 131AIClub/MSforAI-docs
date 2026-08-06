---
title: "计算图自动微分技术"
order: 3
---

# 计算图自动微分技术

## 自动微分

深度学习框架的核心是**自动微分**。这里解释一下自动微分是什么。

我们有一个表达式$f(x)$。它的参数只有$x$。我们想要求$x$的微分，朴素的思路是使用定义法：假设我们想要求$x_0$处的微分（导数值），我们会用下面这个式子近似：

$$
f'(x_0) \approx \frac{f(x_0 + \delta x) - f(x_0)}{\delta x}
$$

当然，$\delta x$要足够小。使用代码实现：

```python
from typing import Callable
def differ(f: Callable[[float], float], x: float, epsilon: float=1e-6) -> float:
    return (f(x + epsilon) - f(x)) / epsilon

def f(x: float) -> float:
    return x ** 2 + 2 * x

print(differ(f, 1.))    # \approx 4
```

通过这种方法，我们可以让计算机估算出每个变量的微分。让计算机可以计算函数的微分，这称之为**自动微分**。

但是这种方法有一定的缺陷。当参数量很大时，逐个计算是很慢的（其实也可以批量计算，但那样又内存昂贵）。并且对于很深的嵌套函数，会有大量重复的计算。因此，现代自动微分并不使用这种技术。

除了定义法之外，我们还可以使用导函数法。我们可以提前计算出函数的导函数。例如，我们知道$f(x) = x^2 + 2x$，我们可以快速计算出其导函数为$f'(x) = 2x + 2$。然后我们就可以代入$x$直接进行计算。

更好的是，我们提到过，一些函数的导函数十分简单。例如 sigmoid 函数与 ReLU 函数。sigmoid 函数可以复用运算结果，ReLU 函数则只需要大小比较操作。因此，现代自动微分技术，采用**导函数法**。

这样，自动微分的关键就在于，如何让计算机自动推导出算式的导函数。

## 计算图

计算图是对算式的一种建模方法，它便于计算机处理。计算图中有两种节点，分别是**数据**和**操作**。边代表输入输出，或者说是依赖关系。下面是一个例子。

算式$f(x, y, z) = x^2 + y^2 + x^2y^3 + z$的计算图为

![](/static/B3IVbj8QxoihI0xh68vch2yWnec.png)

这非常好理解。对于计算机来说，我们使用面向对象的方式，只需要实现 Node 类，随后继承出 Variable 类和 Operation 类即可表示计算图。我们可以使用指针方法来表示节点之间的依赖关系。

我们可以使用运算符重载的方法来捕捉算式构建时的依赖关系，得知计算图的拓扑结构。

## 计算图与链式法则自动微分

对于复杂的算式求导是很难的，就算是人类，可能都很难。但是对于单个操作求导是很简单的。例如乘法求导，结果就是系数；加法求导，结果是 1。而算式可以看作是操作的复合。我们可以使用复合函数求导的链式法则，来进行求导。

对于一个由操作组成的算式来说，对其中一个参数的导数为：

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial op_n} \frac{\partial op_n}{\partial op_{n-1}}
\frac{\partial op_{n-1}}{\partial op_{n-2}}
\cdots
\frac{\partial op_2}{\partial op_1}
\frac{\partial op_1}{\partial x}
$$

我们可以让计算机计算出每个$\frac{\partial op_i}{\partial op_{i-1}}$，这一般是比较简单的。因为我们把操作拆分得很细，往往只是一些加减乘除。随后我们将每一项相乘，即可得到我们需要的结果。

至于操作之间的依赖关系，我们可以利用计算图获得。下面用一个例子来说明。

我们就直接拿计算$f(x, y, z) = x^2 + y^2 + x^2y^3 + z$在$(1, 2, 3)$处的梯度为例。我们将利用链式法则进行求解。

我们观察计算图，从结果向输入参数推导。首先我们需要计算操作$+$的导数。其写出来为：

$$
\begin{aligned}
f &= add(x^2 + y^2 + x^2y^3, z) \\
\frac{\partial f}{\partial z} &= 1 \\
\frac{\partial f}{\partial (x^2 + y^2 + x^2y^3)} &= 1

\end{aligned}
$$

我们可以发现，加法操作对两个输入参数求导，其实就是 1。而我们在运行计算图时，可以将加法操作的输入参数存储起来（而不是运算完之后抛弃）。这样就不需要存储一个很大的式子。只需要存储数值。

现在，我们知道了$\frac{\partial f}{\partial z} = x^2 + y^2 + x^2y^3 + 1$，我们继续求$x$与$y$。

$$
\begin{aligned}
x^2 + y^2 + x^2y^3 &= add(x^2 + y^2, x^2y^3) \\
\frac{\partial(x^2 + y^2 + x^2y^3)}{\partial (x^2 + y^2)} &= 1 \\
\frac{\partial(x^2 + y^2 + x^2y^3)}{\partial x^2y^3} &= 1
\end{aligned}
$$

接下来我们研究$x^2 + y^2$。

$$
\begin{aligned}
x^2 + y^2 &= add(x^2, y^2) \\
\frac{\partial (x^2 + y^2)}{\partial x^2} &= 1 \\
\frac{\partial (x^2 + y^2)}{\partial y^2} &= 1
\end{aligned}
$$

$x^2$与$y^2$都是经过$power$操作得到的：

$$
\begin{aligned}
x^2 &= power(x, 2) \\
\frac{\partial x^2}{\partial x} &= 2x \\
y^2 &= power(y, 2) \\
\frac{\partial y^2}{\partial y} &= 2y
\end{aligned}
$$

别忘记了，我们现在只计算了$x^2 + y^2$路径得到的导数值。还需要计算$x^2y^3$的导数值：

$$
\begin{aligned}
x^2y^3 &= multiply(x^2, y^3) \\
\frac{\partial x^2y^3}{\partial x^2} &= y^3 \\
\frac{\partial x^2y^3}{\partial y^3} &= x^2

\end{aligned}
$$

有关$\frac{\partial x^2}{\partial x}$的导函数，我们已经计算过了。我们不会重复计算。接下来计算$\frac{\partial y^3}{\partial y}$。

$$
\begin{aligned}
y^3 &= power(y, 3) \\
\frac{\partial y^3}{\partial y} &= 3y^2
\end{aligned}
$$

我们手工计算出了所有项。现在我们使用链式法则合并它们

$$
\begin{aligned}
\frac{\partial f}{\partial x} &= \frac{\partial f}{\partial (x^2 + y^2 + x^2y^3)} \cdot (\frac{\partial (x^2 + y^2 + x^2y^3)}{\partial(x^2 + y^2)}\cdot \frac{\partial(x^2 + y^2)}{\partial x^2} + \frac{\partial(x^2 + y^2 + x^2y^3)}{\partial x^2y^3}\cdot\frac{\partial x^2y^3}{\partial x^2})\cdot \frac{\partial x^2}{\partial x}\\
&= 1\cdot(1\cdot1 + 1\cdot y^3)\cdot 2x = 2x(1+y^3)\\
\frac{\partial f}{\partial y} &= \frac{\partial f}{\partial(x^2 + y^2 + x^2y^3)}\cdot(\frac{\partial (x^2 + y^2 + x^2y^3)}{\partial(x^2 + y^2)}\cdot
\frac{\partial(x^2 + y^2)}{\partial y^2}\cdot \frac{\partial y^2}{\partial y} + \frac{\partial(x^2 + y^2 + x^2y^3)}{\partial x^2y^3}\cdot\frac{\partial x^2y^3}{\partial y^3}\cdot\frac{\partial y^3}{\partial y})\\
&= 1\cdot(1\cdot 1\cdot 2y + 1\cdot x^2\cdot 3y^2) = 2y+3
x^2y^2\\
\frac{\partial f}{\partial z} &= x^2 + y^2 + x^2y^3 + 1\\
\nabla f &= \begin{bmatrix}
2x(1+y^3) \\
2y+3x^2y^2 \\
1
\end{bmatrix}
\end{aligned}
$$

将$(1, 2, 3)$代入，得到：

$$
\nabla f = \begin{bmatrix}
18 \\
16 \\
1
\end{bmatrix}
$$

我们是手工计算，一直采用符号推理。但是计算机实现时，实际上只保存了数值。计算机并不是先把每个微分项推导出来，再全部乘起来，而是一边推导一边乘。

在实际实现的过程中，算法会对计算图进行**反向传播**，按照一定顺序遍历计算图。不断累乘梯度值，当遍历完成时，也就求出了每个参数的导数值。
