---
title: "NumPy 语法入门"
order: 2
---

# NumPy 语法入门

NumPy 的语法尽量贴合 Python 原版的列表。在这里，我们先介绍基本的 NumPy 对象：NdArray。

## NdArray

ndarray 是 numpy 中用于表示向量，矩阵，张量的类型。其存储在一段连续的内存上。定义 ndarray 有许多种方法。这里先使用最简单的。

```python
import numpy as np
vec = np.array([1, 2, 3, 4])
mat = np.array([[1, 2, 3],
                [4, 5, 6]])
print(vec)
print(mat)
```

定义时，可以传入 `dtype` 参数，用于指定存储类型。与列表不同，ndarray 中必须全部元素**是相同类型**。如果不传入该参数，类型将自动判断。

以下是可用的参数（这里只列出 numpy 的标准类型，我们也推荐使用这种方式定义类型）：

| **对象** | **含义** |
| --- | --- |
| `np.int8` | 8位有符号整数 |
| `np.int16` | 16位有符号整数 |
| `np.int32` | 32位有符号整数 |
| `np.int64` | 64位有符号整数 |
| `np.uint8` | 8位无符号整数 |
| `np.float32` | 单精度浮点数 |
| `np.float64` | 双精度浮点数 |
| `np.bool_` | 布尔类型 |
| `np.complex64` | 复数 |
| `np.str_` | 长度为 10 的unicode字符串 |
| `np.bytes_` | 长度为 10 的字节字符串 |

下面是一个类型定义示例：

```python
import numpy as np
vec = np.array([1, 2, 3, 4], dtype=np.int8)
mat = np.array([[1, 2, 3],
                [4, 5, 6]], dtype=np.float32)
print(vec)
print(mat)
```

在所有类型中，我们最经常使用的是 `float32`。

### 定义特定 NdArray

有时候，我们需要定义一些特殊的 array，例如全是 0，全是 1，或者是我们根本不关心数据，我们只想要一块特定形状的 array，用来存放中间结果之类的。

定义全 0 的 array，我们使用 `np.zeros` 函数：

```python
import numpy as np
arr = np.zeros((2, 3, 4), dtype=np.float32)
print(arr)
```

定义全 1 的 array，我们使用 `np.ones` 函数：

```python
import numpy as np
arr = np.ones((4, 3, 2), dtype=np.uint8)
print(arr)
```

我们不关心具体的值，只希望要一个特定的形状，可以使用 `np.empty` 函数。这可以减少赋值的开销。

```python
import numpy as np
arr = np.empty((3, 4, 5), dtype=np.float32)
print(arr)
```

### NdArray 索引访问

ndarray 的索引访问几乎与嵌套列表是相同的，但是也有一些区别。

```python
import numpy as np
vec = np.array([1, 2, 3, 4, 5], dtype=np.float32)
vec[1] = 0.1
print(vec)

mat = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(mat[1])
print(mat[1][2])
print(mat[1, 2])
print(mat[:, 1])

tensor = np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    [[-6, -5, -4],
     [-3, -2, -1]]
], dtype=np.float32)
print(tensor[1, :, 1])
```

`:` 的含义是“全取”，以此来实现更加灵活的访问。

### NdArray 的计算

首先是基本的线性代数计算，NumPy 肯定是支持的。

```python
import numpy as np
mat1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
mat2 = np.array([[-1, -2], [-3, -4], [-5, -6]], dtype=np.float32)
mat3 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32)

print(mat1.shape, mat2.shape, mat3.shape)
print(mat1 + mat3)
print(mat1 - mat3)
print(mat1 * mat3)
print(mat1 / mat3)
print(mat1 @ mat2)    # np.matmul(mat1, mat2)
```

numpy 还支持求最大/小值，求最大/小值索引：

```python
import numpy as np
tsr = np.random.rand(2, 3, 4)
tsr[0][1][2] = 100
print(np.max(tsr))
print(np.argmax(tsr))
```

numpy 中可以方便转置。转置操作是很便宜，因为它只会修改元信息。

```python
a = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
print(a)
print(a.shape)

b = a.T  # 转置，shape (3, 2)
print(b)
print(b.shape)

# 转置只是修改了 strides 和 shape 的读写顺序，不涉及数据拷贝
# 可以验证：a 和 b 共享同一块内存
a[0, 0] = 100
print(b[0, 0])  # 输出 100，说明确实共享内存
```

## Broadcast 广播机制

Broadcast 广播机制是一个非常重要的机制。其基本机制十分简单。

我们知道，ndarray 的加减乘除都是要求形状匹配。例如对应位置加减乘除需要两个数组形状一样，矩阵乘法要求最后两个维度满足矩阵乘法的形式，前面的维度要一致。但是假如我们想要实现这样一个操作：向量$\mathbf{v}\in \R^n$是个行向量，矩阵$\mathbf{M}\in \R^{m\times n}$，我们想让$\mathbf{M}$的每一行加上$\mathbf{v}$。我们可能会使用 repeat 方式，将$\mathbf{v}$复制成一个矩阵，它每一行都是$\mathbf{v}$，然后再相加。

但是这样其实不太好。因为它占用了更多的存储空间，拷贝也会有开销。实际上我们可以让计算机直接做这件事情。NumPy 为了让用户可以实现这样的操作，提供了广播机制。

当一个操作的输入数组形状不匹配时，numpy 会进行广播机制判定，它的规则如下：

- 将所有输入数组的维数补全到维数最大的那个数组。补全的方式为在前面加 1。例如，三个输入数组的形状分别为(2, 3)，(2, 3, 4)，(2, 3, 4, 5)，则会被补全成(1, 1, 2, 3)，(1, 2, 3, 4)，(2, 3, 4, 5)。
- 检查每个输入数组的维度。如果维度大小是相同的，则没事。如果维度大小不同，并且其中一个数组的维度是 1，则触发广播机制，也是合法的。对应维度大小为 1 的那个数组将在该维度上被广播。否则输入不合法。例如形状(1, 3, 4)和(5, 3, 1)是合法的一组输入，输入形状(1, 2, 3)与(3, 2, 2)不合法。

广播机制实际上不会对数据进行复制，因此效率比拷贝方法高，而且内存开销也小。

## 多查资料多问 AI

NumPy 中的细节真的非常多。而且目前的教育中，有关矩阵的讨论是很少的。使得多维数组的操作实际上是一门学问，并且大家比较欠缺这方面的经验。课程中不可能全部涉及到。甚至说，这里提到的相关知识，只是 NumPy 最基础的一些概念，而你往往需要进行一些不太寻常的多维数组操作，你就需要去寻找 numpy 是否提供了一些接口，你如何使用这些接口来达成你的目的。并且，这个部分的很多东西思考起来是很反人类的。

这就需要大家多去查资料，多去问 AI。很多细节并不复杂，是一个知不知道的问题。
