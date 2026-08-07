---
title: "NumPy"
order: 4
index: "04"
label: "COMPUTING"
description: "用数组表达向量与矩阵，理解批量计算如何连接数学和程序。"
---

# 第三章 NumPy，使用计算机进行线性代数计算

在第一章中，我们介绍了线性代数表示的神经网络计算。而在第二章我们又学习了 Python。大家都说 AI 是用 Python 写的。那很好，我们现在开始用 Python 写神经网络吧！

根据一点点程序设计的基本思想，我们肯定希望类似矩阵乘法，矩阵加法的操作封装好，封装成函数，最好还可以把向量，矩阵用类封装（面向对象）。不过，肯定有库实现了这些东西。但在这里，为了课程的继续，我们还是尝试自己实现一下。这里直接用列表表示向量，嵌套列表表示矩阵：

```python
from typing import List
def matmul(mat1: List[List[float]], mat2: List[List[float]]) -> List[List[float]]:
    r1, c1 = len(mat1), len(mat1[0])
    r2, c2 = len(mat2), len(mat2[0])
    assert c1 == r2    # 形状检查
    result = [[0. for _ in range(c2)] for _ in range(r1)]
    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result

mat1 = [
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
]

mat2 = [
    [0.1, 0.4, 0.2],
    [0.2, -0.5, 0.7],
    [-0.1, -0.2, -0.3]
]

print(matmul(mat1, mat2))
```

但是，我们可以看出来，这个实现非常麻烦。我们需要自己检查形状等等。最关键的是，这个实现效率是极低的！以一个输入层大小 784，隐藏层大小 2048，输出层大小 10 的 MLP 举例，最大的矩阵乘法要处理一个 $(\text{batch\_size}, 784)$ 和 $(784, 2048)$ 大小的矩阵相乘。这在现代处理器上其实也没有什么压力。我们跑一个测试（假设 $\text{batch\_size}$ 是 32）：

```python
import time
batch_size: int = 32
mat1 = [[0.1 for _ in range(784)] for _ in range(batch_size)]
mat2 = [[0.1 for _ in range(2048)] for _ in range(784)]
t1 = time.time()
matmul(mat1, mat2)
t = time.time() - t1
print(t)
```

这在我的机器上花费将近 3 秒(2.8159382343292236)！这太慢了。这还只是一个玩具级的网络（即使对我的 CPU 来说，也是玩具级的）。造成这种缓慢的问题有很多，感兴趣的同学可以查询相关资料。这里我们解释为 Python 作为解释器速度天然有劣势，并且需要维护很多对象，列表本身是个动态还不连续的结构，并且 Python 存在 GIL（全局解释器锁）使得程序只能单线程。

于是，有人（一群人）专门为 Python 实现了一个高性能的线性代数库 NumPy。NumPy 是由 C 编写的，进行了很强的优化，然后暴露接口给 Python 调用。

虽然还没介绍 NumPy 的语法，但是这里对比一下时间，给大家看一下性能差距：

```python
import numpy as np
batch_size: int = 32
# 随机生成两个矩阵
mat1 = np.random.rand(batch_size, 784)
mat2 = np.random.rand(784, 2048)

import time
t1 = time.time()
for _ in range(1000):    # 这里因为numpy太快了所以测1000次求平均
    np.matmul(mat1, mat2)
t = time.time() - t1
print(t/1000)
```

在我的机器上，这个时间是 0.0015103209018707275 秒，快了 1866 倍！

> 实际情况下快不了这么多，这里 1000 次测试缓存命中率太高了。
