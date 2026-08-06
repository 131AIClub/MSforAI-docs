---
title: "NumPy 实现线性回归波士顿房价"
order: 3
---

# NumPy 实现线性回归波士顿房价

## 线性回归

我们在高中都学习过线性回归方法。这里介绍其更加通用的形式，即多元的情况。

我们的场景中存在 $n$ 个输入变量 $x_i$ 与 1 个预测值 $\hat{y}$ 。对于预测值，它满足：

$$
\hat{y} = \sum_{i=1}^n w_ix_i + b
$$

写成线性代数形式：

$$
\hat{y} = \mathbf{wx}^\top + b
$$

其中，$\mathbf{w}$ 与 $\mathbf{x}$ 都是行向量。

为了方便计算，这里引入一些概念。我们先将$\mathbf{w}$与$b$合并到一起。$\mathbf{w}$中添加一个元素，其值为$b$，而$\mathbf{x}$中添加一个值，其值始终为 1。长这个样子：

$$
\mathbf{w} = \begin{pmatrix}
b & w_1 & w_2 & w_3 & \cdots & w_n
\end{pmatrix};\;
\mathbf{x} = \begin{pmatrix}
1 & x_1 & x_2 & x_3 & \cdots & x_n
\end{pmatrix}
$$

我们把所有数据集中样本特征拼成一个矩阵，称之为**设计矩阵** $\mathbf{X}\in \R^{N\times (n+1)}$ ，所有真实值组成**标签向量** $\mathbf{y}\in \R^N$ 。

预测值可以写为：

$$
\hat{\mathbf{y}} = \mathbf{wX}^\top
$$

我们使用均方损失，尝试找到最优参数 $\mathbf{w}^*$ ：

$$
loss = (\mathbf{\hat{y}} - \mathbf{y})(\mathbf{\hat{y}} - \mathbf{y})^\top = (\mathbf{wX}^\top-\mathbf{y})(\mathbf{wX}^\top-\mathbf{y})^\top
$$

微分：

$$
\begin{aligned}
\mathbf{d}loss &= \mathbf{d}[(\mathbf{wX}^\top-\mathbf{y})(\mathbf{wX}^\top-\mathbf{y})^\top] \\
&= 2(\mathbf{wX}^\top-\mathbf{y})\mathbf{d}(\mathbf{wX}^\top-\mathbf{y})^\top \\
&= 2(\mathbf{wX}^\top-\mathbf{y})[\mathbf{d}(\mathbf{Xw}^\top)-\mathbf{dy}^\top] \\
&= 2(\mathbf{wX}^\top-\mathbf{y})[\mathbf{d(X)w}^\top + \mathbf{Xdw}^\top-\mathbf{dy}^\top]
\end{aligned}
$$

有关 $\mathbf{w}$ 的微分项：

$$
2(\mathbf{wX}^\top-\mathbf{y})\mathbf{Xdw}^\top
$$

因此，$loss$ 对 $\mathbf{w}$ 的梯度为：

$$
\nabla_\mathbf{w}loss = 2(\mathbf{wX}^\top-\mathbf{y})\mathbf{X}
$$

令梯度为 $\mathbf{0}$（类似导数为 0 求函数极值）：

$$
\begin{aligned}
2(\mathbf{wX}^\top-\mathbf{y})\mathbf{X} &= \mathbf{0} \\
\mathbf{wX}^\top\mathbf{X-yX} &= \mathbf{0} \\
\mathbf{wX}^\top \mathbf{X} &= \mathbf{yX} \\
\mathbf{w} &= \mathbf{yX}(\mathbf{X}^\top\mathbf{X})^{-1}
\end{aligned}
$$

我们得到了在均方损失意义下最好的参数：

$$
\mathbf{w^*} = \mathbf{yX}(\mathbf{X}^\top\mathbf{X})^{-1}
$$

## 波士顿房价

波士顿房价是一个很经典的机器学习任务。波士顿房价的任务如下：

我们都知道房价肯定和附近基础设施，距离城市中心的距离等因素有关，但是我们在生活中一般是定性分析的。但是我们想要更加详细地研究房价问题。有人统计了波士顿附近多个房子的相关属性，每个房子收集十三个可能影响房价的特征。特征如下：

| **特征名称** | **含义** |
| --- | --- |
| CRIM | 城镇人均犯罪率 |
| ZN | 占地面积超过25,000平方英尺的住宅用地比例 |
| INDUS | 城镇非零售商业用地比例 |
| CHAS | 是否靠近查尔斯河（0代表不靠近，1代表靠近） |
| NOX | 一氧化氮浓度 |
| RM | 每栋住宅的平均房间数 |
| AGE | 1940年之前建成的自用住房比例 |
| DIS | 到波士顿五个就业中心的加权距离 |
| RAD | 径向高速公路可达性指数 |
| TAX | 每10,000美元的全额财产税率 |
| PTRATIO | 城镇师生比例 |
| B | 城镇黑人比例 |
| LSTAT | 低收入人口比例 |

我们的目标变量是 MEDV，含义是自住房屋的中位数价值。

## 任务分析

我们将使用线性回归方法来预测 MEDV。首先我们定义线性回归模型的输入，即特征向量：

$$
\mathbf{x} = \begin{pmatrix}
1 & CRIM & ZN & INDUS & \cdots & LSTAT
\end{pmatrix}
$$

我们定义模型的参数：

$$
\mathbf{w} = \begin{pmatrix}
b & w_1 & w_2 & w_3 & \cdots & w_n
\end{pmatrix}
$$

我们将数据集中所有样本的 MEDV 值组织成标签向量：

$$
\mathbf{y} = \begin{pmatrix}
MEDV_1 & MEDV_2 & MEDV_3 & \cdots & MEDV_n
\end{pmatrix}
$$

所有样本的特征向量组织成设计矩阵：

$$
\mathbf{X} = \begin{bmatrix}
\mathbf{x}_1 \\
\mathbf{x}_2 \\
\mathbf{x}_3 \\
\vdots \\
\mathbf{x}_N
\end{bmatrix}
$$

我们就可以直接计算出最优参数 $\mathbf{w^*}$ 。

$$
\mathbf{w^*} = \mathbf{yX}(\mathbf{X}^\top\mathbf{X})^{-1}
$$

## 代码实现与分析

实现线性回归模型类。

这里我们先实现一个线性回归类：

```python
import numpy as np

class LinearRegression:
    def __init__(self, n: int) -> None:
        self.n = n
        self.w = np.empty(n+1, dtype=np.float32)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[1] == self.n + 1
        return np.matmul(self.w, X.T)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.w = labels @ features @ np.linalg.inv(features.T @ features)
```

我们需要加载波士顿房价数据集。由于一些不能说的原因，波士顿房价数据集无法在新版本的 `sklearn` 库中直接导入。我们准备好了波士顿房价的数据集，从本地导入。

```python
FILE = r'./boston_data.csv'
boston_data = np.loadtxt(FILE, delimiter=',', skiprows=1, dtype=np.float32)
print(boston_data.shape)
```

这个 csv 文件是这样组织的：

![](/static/MNbnblOwsoYroUxWovxc433UniF.png)

我们跳过了第一行，因为第一行是字符，无法转化为 `np.float32`。

接下来，我们分离特征与标签，训练集与测试集。这里测试集按照 20% 划分。

```python
SAMPLE_NUM = boston_data.shape[0]
TRAIN_SAMPLE_NUM = int(SAMPLE_NUM * 0.8)

train_features = boston_data[:TRAIN_SAMPLE_NUM, :13]
train_labels = boston_data[:TRAIN_SAMPLE_NUM, 13]
print(train_features.shape, train_labels.shape)

test_features = boston_data[TRAIN_SAMPLE_NUM:, :13]
test_labels = boston_data[TRAIN_SAMPLE_NUM:, 13]
print(test_features.shape, test_labels.shape)
```

注意，特征向量需要在最前面补 1，进行特征扩展。

```python
train_features = np.concatenate((np.ones((TRAIN_SAMPLE_NUM, 1), dtype=np.float32), train_features), axis=1)
test_features = np.concatenate((np.ones((SAMPLE_NUM - TRAIN_SAMPLE_NUM, 1), dtype=np.float32), test_features), axis=1)
print(train_features.shape, test_features.shape)
print(train_features[:5])
print(test_features[:5])
```

然后是拟合部分的代码

```python
model = LinearRegression(13)
model.fit(features=train_features, labels=train_labels)
predict = model(test_features)
```

使用 `matplotlib` 库进行图像绘制：

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(test_labels, label='True Value', color='r', alpha=0.7)
plt.plot(predict, label='Prediction', color='b', linestyle='--', alpha=0.7)
plt.title('Boston Housing: Prediction vs True Value')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()
```

结果如下：

![](/static/MUfBbBdCnoEIFfxXZX3c0H9dnqg.png)

可以看到效果还行。
