# MS for AI 讲义 副本

# 序章

本讲义是由 131AIClub(东南大学人工智能协会)开设的人工智能入门系列课程 MS for AI 配套讲义.

[这里填课程视频]

[鸣谢等内容]

## 是什么

MS for AI, 全称为"Missing Semester for Artificial Intelligence", 即"人工智能教育中缺失的一课". 本课程旨在补全本科教育中有关人工智能教育的缺失部分. 本科教育重理论, 轻实践, 并且缺乏领域前沿知识. 因此本课程重实践, 并且会涉及更多与业界正在使用的技术相关的知识.

本课程难度低于"动手学深度学习". 相比其更加入门, 门槛更低.

## 讲什么

本课程将从 AI 领域的总览开始, 先讲解 AI 领域的历史与发展, 现代的技术栈, 形成整体的观念;第二, 三章将讲解 AI 开发所需的基本技能, 如 Python 基础, 张量计算等; 第四章将讲解机器学习任务的基本组成, 从较概括的视角描述 AI; 第五章将介绍深度学习框架 PyTorch, 同时也会补全更多有关深度学习的理论; 第六, 七章分别介绍深度学习在计算机视觉与自然语言处理中的运用; 第八章将详细讲解 Transformer 架构; 第九章将讲解大型语言模型的基本原理, 训练流程, 在这个过程中将补全一点强化学习知识; 第十章将讲解多模态技术.

客观来说, 本课程较为**功利**. 基本只挑选了最有用的知识. 这也是因为传统机器学习的大部分方法, 同学都可以在课内学习到. 而编程技巧, 在课内则缺乏锻炼. 例如 Numpy 与 PyTorch 编程, 对推理框架的理解等.

但是本课程仅仅为**入门**课程, **缺乏大量数学原理和细节**. 这些都需要同学根据兴趣, **自己深入研究**. 本课程更多起到入门与拓宽视野作用.

## 怎么学

本课程面向几乎 0 基础的同学. 想要比较扎实地完成本课程, 你最好需要:

- 坚持.
- **自己搜寻资料, 学习, 然后解决问题的能力**. 对于没有接触过 AI 的同学来说, AI 的一些理论理解起来是比较花时间的, 而且理论到实践的 gap 也非常大. 这期间一定会出现大量本课程无法涵盖, 社区也无法及时回答的问题, 这就需要同学可以自己解决.
- 一点点计算机基础(最好是上过计算机通识课程, 掌握基本程序设计思想, 熟悉终端与命令行等)

课程的作业最好还是自己完成一下. 对于很多同学来说, 解决问题的过程中培养的**自主解决问题的能力**其实才是最大的收获.

# 第一章 Overview

## AI 是什么?

大家应该之前或多或少听过很多了. 就算没专门查过和看过, 也大概有"人工智能就是让电脑像人一样"的模糊感觉. 毕竟这个名词非常"恰如其名", 就是"人类制造的智能".

感性地说, **人工智能是研究和制造能够模仿, 扩展人类智能的系统**.

注意, 当我们在考察系统的智能时, 我们是否在意它的结构?

假如说, 我们有一个程序, 它很简单, 它会把输入句子中的"吗?"替换成"!", 把"你"替换成空, 把"我"替换成"你". 你会认为这个程序有智能吗?

```cpp
人: 在吗?
程序: 在!
人: 你好吗?
程序: 好!
人: 我做得好吗?
程序: 你做得好!
```

我们很明显**不认为**它有智能. 但是或许在某些特定情况下, 真的可以骗到人一小下. 这还仅仅只有 3 条规则. 如果我们有非常多的规则, 上万条乃至上亿条, 你还能确定你不会在一无所知的情况下被程序骗到吗?

但是你会认为这种基于规则的结构拥有智能吗? 我想一些人会认为"这种结构怎么能说是智能呢?" 但应该还是有一些人觉得"难说!", 因为**规则够多够复杂, 说不定就有智能了**, 毕竟我们大脑, 单拿出来一个神经元也挺简单的.

这个问题确实很难说. 不过如果我们认为系统的结构本身影响其是否是智能, 会影响我们理论的发展. 因为我们必须考虑"什么样的结构是智能"这个超难的问题. 相比之下, 只考虑"系统的表现像不像智能", 就显得简单多了.

所以, 至少**在大多数讨论情景下**, 我们都认为"是否具有智能"这件事, 要从系统的表现来看, 而与系统的内部结构无关.

在这个前提下, 我们引入机器学习视角下的智能:

定义输入空间$I = \{ i_1, i_2, i_3, \cdots\}$与输出空间$O = \{ o_1, o_2, o_3, \cdots \}$, 映射集合$F = \{f | f: I \rightarrow O \}$. 定义一个性能评价函数$Critic$. $Critic(f)$的值是一个标量, 用于表示映射$f$的好坏. 机器学习任务, 就是在$F$里寻找使得$Critic$最好的映射.

如果这个性能评价函数被我们设计成了"我们认为符合智能的样子", 通过机器学习方法, 就可以去寻找这个最好的映射. 例如, 我们的输入空间是**全世界所有的图片组成的集合**, 我们的输出空间是这张图片里的是猫还是狗, 我们的性能评价函数就是映射对所有图片分类结果的**正确数量**, 那么我们寻找最大正确数的映射的过程, 就是在寻找我们认为的"可以分类猫和狗的智能".

在大多数问题中, 由于输入和输出空间很大, 映射空间也很大, 我们只能采样, 获取样本, 然后将映射参数化, 使用数理统计方法估计映射的参数.

## AI 的发展历程

施工中

## 深度神经网络

### 神经元

![](static/LMzxbUGCcoklqoxrqYxcjhdBnYf.png)

深度神经网络的基本组成单位是**神经元**. 传统意义的神经元有多个输入和一个输出.

一个神经元有多少个输入, 就有多少个**权重**. 权重是个标量值, 用于表示联系的紧密程度.

假如说一个神经元有$n$个输入, 分别为$x_1, x_2, x_3, \cdots, x_n$; 对应的权重分别为$w_1, w_2, w_3, \cdots, w_n$. 则神经元的输出$y$为:

$$
y=w_1x_1+w_2x_2+w_3x_3 + \cdots + w_nx_n
$$

实际上就是**加权求和.**

我们会发现, 神经元的输出, 对于任意一个输入来说都是线性的. 我们为了提高神经元的表示能力, 可以进行一个非线性映射. 这个映射我们称之为**激活函数**. 这里用$f$表示:

$$
y=f(w_1x_1+w_2x_2+w_3x_3 + \cdots + w_nx_n)
$$

常见的$f$有 **sigmoid 函数**, **tanh 函数**与 **ReLU 函数**.

[施工, 介绍一下这三个函数]

这样神经元就比较完整了. 不过我们还希望神经元拥有对自己的输入进行整体平移的能力, 这就引入了**偏置 bias**, 我们一般用$b$表示:

$$
y=f(w_1x_1+w_2x_2+w_3x_3 + \cdots + w_nx_n + b)
$$

这样的神经元一共有$n+1$个可学习参数, 分别是$n$个权重值$w_i$和 $1
$个偏置值$b$.

### 线性神经网络层

![](static/BAuHb8fzGoz7fpxIGpZcUlbznPh.png)

将$m$个神经元排成一列, 每个神经元共享输入, 因此有$n$个输入, $m$个输出.

我们把输入组织成一个$n$维向量:

$$
\bold{x} = 
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_n
\end{bmatrix}
$$

其中, 每一行代表一个神经元的输入. 我们也把权重组织成一个$m\times n$的矩阵:

$$
\bold{W} = 
\begin{bmatrix}
w_{11} & w_{12} & w_{13} & \cdots & w_{1n} \\
w_{21} & w_{22} & w_{23} & \cdots & w_{2n} \\
w_{31} & w_{32} & w_{33} & \cdots & w_{3n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & w_{m3} & \cdots & w_{mn}
\end{bmatrix}
$$

其中每一行代表一个神经元的所有权重值. 我们把偏置表示成一个$m$维向量:

$$
\bold{b}=
\begin{bmatrix}
b_1 \\
b_2 \\
b_3 \\
\vdots \\
b_m
\end{bmatrix}
$$

每个神经元都进行**相同的计算**, 因此我们可以把线性神经网络层的计算用下面的线性代数操作表示:

$$
\bold{y} = f(\bold{Wx+b})
$$

这里的$f$是个**向量值函数**, 它的效果就是对每个值施加之前的非线性变换.

> 你可能会在别的地方看到这么写:
>
> $$
> \bold{y} = f(\bold{xW^T+b})
> $$
>
> 为什么$\bold{W}$需要转置? 这是因为在实际实现中, $\bold{x}$是个矩阵, 它的形状为$batch\_size \times n$. $n$很好理解, 而$batch\_size$你可以暂时理解成, 为了效率, 我们需要批量处理数据, 要把很多个输入数据拼起来, 向量就被拼成了矩阵.
> 而且这里为了好看, 还把$\bold{x}$的形状从 $1 \times n$改成了$n\times 1$. $\bold{b}$ 的形状也要改过来.

### 多层感知机 MLP

![](static/DYmUbVf1CoLj6zxGV8VcsQvIndf.png)

将多个线性神经网络层直接前后连接(即前一层的输出是后一层的输入), 就构成了多层感知机 MLP.

一般的 MLP 图中的第一层其实是输入层, **不算做层数里面**, 只是提供输入值. MLP 的关键参数设置是**每层的神经元数**量. 理论上已经证明, 只要足够大, 两层的 MLP 就可以以**任意要求精度拟合任意任务**.

### 神经网络的参数优化

神经网络拥有一些(大多数情况下是大量)可学习参数. 但是网络并不是平白无故就有了一组能解决问题的好参数. 我们并不知道面对一个任务时, 神经网络到底该采用什么参数. 但是神经网络可不是$f(x) = ax + b$这种简单的东西. 它过于复杂, 我们连初等形式都很难写出来(这往往涉及线性代数和递归).

神经网络的参数寻找似乎很困难, 不过这里我们先考虑一个更加通用的场景：

假如说我们有一个带参函数$f_\theta$, $\theta$就是参数. 为了方便理解, 假如说这个函数就是$f_\theta(x) = ax^2 + bx + c$, $\theta$就是$(a, b, c)$三元组. 修改$\theta$就是在修改$f$, 虽然$f$的形式不变.

现在, 我给你一组有序对$D = \{(x_i, y_i)\}_N$, $N$是有序对数量, 请你找到最符合这些有序对的$\theta$.

在高中, 我们会利用残差平方和最小来直接算出这个参数. 在$f$的形式不是很复杂的时候, 我们确实可以利用这种方法来求出参数的解析解. 我们可以从这个寻找最优参数的过程中总结出一个经验:

- 首先, 我们需要找到一个衡量$f$有多好的方法. 在这里, 我们选择了残差平方和函数的值.
- 然后, 参数寻找问题变成了求函数极值点的问题, 即将$\theta$看作变量, 求残差平方和函数的极小值点.

神经网络参数优化, 也使用相同方法. 我们把神经网络看作一个函数$Net_\theta$, $\theta$表示了神经网络的参数. $\bold{x}$是神经网络的输入, 神经网络的计算过程表示为$Net_\theta(\bold{x})$. 神经网络的计算过程, 称之为"**前向传播**", 非常形象, 即信号(数据)在网络中从前往后(从浅向深)传播. "**推理**"这个术语也可以用于指代神经网络的前向传播.

为了让参数有个优化的依据, 我们需要**数据集**. 数据集是从环境中采样得到的样本集合. 一般来说会带有一定的随机性和方差. 每个样本都由两部分组成: **特征(feature)**与**标签(label)**. 特征即样本的一些属性, 标签则是对应样本的真实值. 例如, 你想要构建一个天气预测的数据集. 你统计了几百天的天气, 特征就是每天的属性(或者前一天的属性, 如果是为了预测的话), 例如温度, 湿度, 风速, 云的形状等, 标签就是每天的真实天气, 如晴天, 多云, 小雨等.

这里我们将数据集表示为$Dataset = \{(\bold{x}_i, \bold{y}_i)\}^N_{i=1}$. 神经网络前向传播即为$\bold{y} = Net_\theta(\bold{x})$. 对于数据集中的每个$\bold{x}_i$, 网络都可以计算(预测)出一个$\hat{\bold{y}}_i$. 最后, 我们求和所有残差的平方和$\sum_{i=1}^N{(\bold{y}_i-\hat{\bold{y}}_i)^2}$. 我们的目标就是最小化这个残差平方和. 在这里, 我们把这个衡量模型好坏的函数, 叫做**损失函数(loss function)**. 即:

$$
loss = \sum_{i=1}^N{(\bold{y}_i - \hat{\bold{y}}_i)^2}
$$

损失函数肯定是有关 $\theta
$的函数. 不过$\hat{\bold{y}}_i$的形式有点难写出. 这就导致我们很难像二次/线性函数那样直接写出最好的参数$\theta$ 的表达式. 这里为了找出这个参数, 我们介绍一种优化方法: **梯度下降法**.

#### 梯度与梯度下降

高数里其实对梯度也有比较多的介绍了. 简单的理解可以认为梯度是导数在多维情况下的扩展. 其直观含义是**函数值的增大方向**. 即每个参数应当怎么变化, 才能使得该点处函数值最快速度增大.

梯度的非常不严谨讲解:

我们先引入**雅可比矩阵**. 一个函数有$n$个参数(arguments)$x$, 和$m$个值(values)$f_i$, 则它的雅可比矩阵为:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \frac{\partial f_1}{\partial x_3} &
\cdots \frac{\partial f_1}{\partial x_{n-1}} & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \frac{\partial f_2}{\partial x_3} &
\cdots \frac{\partial f_2}{\partial x_{n-1}} & \frac{\partial f_2}{\partial x_n} \\
\frac{\partial f_3}{\partial x_1} & \frac{\partial f_3}{\partial x_2} & \frac{\partial f_3}{\partial x_3} &
\cdots \frac{\partial f_3}{\partial x_{n-1}} & \frac{\partial f_3}{\partial x_n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_{m-1}}{\partial x_1} & \frac{\partial f_{m-1}}{\partial x_2} & \frac{\partial f_{m-1}}{\partial x_3} &
\cdots \frac{\partial f_{m-1}}{\partial x_{n-1}} & \frac{\partial f_{m-1}}{\partial x_n} \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \frac{\partial f_m}{\partial x_3} &
\cdots \frac{\partial f_m}{\partial x_{n-1}} & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

比较好的事情是, 我们的损失函数一般来说是个标量函数, 即输出只有一个值. 因此在我们的场景下, 雅可比矩阵应该是个向量, 此时, 我们称之为**梯度(而不是雅可比矩阵)**:

$$
\nabla_\Theta loss = \begin{pmatrix}
\frac{\partial loss}{\partial \theta_1} & \frac{\partial loss}{\partial \theta_2} & \frac{\partial loss}{\partial \theta_3} & \cdots & \frac{\partial loss}{\partial \theta_{n-1}} & \frac{\partial loss}{\partial \theta_n}
\end{pmatrix}^\bold{T}
$$

如果我们可以计算出损失函数对网络参数的梯度, 至少我们可以知道在某点, 参数往哪个方向变化, 函数值减少(相比于增大只是乘个-1 即可, 即负梯度)最快. 我们采用贪心策略, 每次都只往函数值减小最快的方向前进一小步. 随着迭代步数的增加, 我们期望我们可以走到函数的最小值位置.

下面用一个简化的例子来形象说明这个过程.

![](static/UkbmbLNtuoZXTJxdHk0cQxllnrg.png)

图中是一个二元函数, 从图中可以直观看出函数的形状. 我们通过梯度下降法寻找最低点的过程, 就是图中的黑色轨迹. 我们每一次都向当前位置函数下降最快的方向前进一点点, 最后形成的就是这样一条"下山"路径.

假设我们在某点处的梯度为:

$$
\nabla J(\theta) = \nabla_\theta J(\theta)
$$

则$\theta$的更新值为:

$$
\delta \theta = -\alpha \cdot \nabla_\theta J(\theta) \\
\theta' = \theta + \delta\theta
$$

$\alpha$是一个小正数(0.01, 0.005 等), 称之为**学习率(learning rate)**.

我们如果采用梯度下降法进行神经网络参数的优化, 就需要有一种计算神经网络中每个参数梯度的办法. 其实, 神经网络每一层的计算是比较简单的, 而是神经网络每一层看作一个独立的函数, 其计算过程就是一个递归的嵌套函数:

$$
Net(\bold{x}) = Layer_n(Layer_{n-1}(Layer_{n-2} \cdots Layer_2(Layer_1(\bold{x}))\cdots))
$$

我们对参数求梯度(求导), 可以运用链式法则. 假设我们想要求损失函数对第$k$层的某个参数$\theta_k^m$的导数:

$$
\frac{\partial loss}{\partial\theta_k^m} = \frac{\partial loss}{\partial \hat{\bold{y}}} \cdot \frac{\partial \hat{\bold{y}}}{\partial Layer_n} \cdot \frac{\partial Layer_n}{\partial Layer_{n-1}} \cdots \frac{\partial Layer_{k+1}}{\partial Layer_k} \cdot \frac{\partial Layer_k}{\partial \theta_k^m}
$$

这个式子简化了一些, 没有拆分到更细的微分式. 不过, 总归是有办法计算的. 我们可以就这么求出全部参数的导数, 我们就得到了梯度. 并且, 这个计算过程是一层一层的(后面的层可以复用前面计算的结果), 从后往前(由深及浅), 计算梯度并进行参数更新的过程也被称为**反向传播**.

因此, 我们在构建网络的过程中, 会慎重考虑组件的数学性质, 尤其是求导的便利性. 例如, sigmoid 激活函数被我们选择, 不只是因为它能将输入压缩(而且压缩后相对大小不变)到$(0, 1)$区间, 而且还因为它求导很简单. 它的求导为:

$$
\frac{\bold{d}f}{\bold{d}x} = f(x)\cdot [1 - f(x)]
$$

我们在前向传播的过程中, 保留计算 sigmoid 函数的值$f(x)$, 这样在反向传播时, 只需要计算简单的减法和乘法即可. 没有复杂的微分过程.

但是 sigmoid 函数也有自己的问题: 当$x$在比较大或者比较小的值时, 其导数很小. 这在一层时似乎不是什么问题, 但在链式求导的深层中, 层层相乘导致梯度数值变得相当小, 深层参数更新不动. 这被称为**梯度消失**问题. 与之对应的, 有些激活函数存在导数值大于 1 的情况(如 ReLU), 在层层相乘的情况下, 梯度的数值会逐渐变得越来越大. 变大有两个坏处, 第一, 当数值大到学习率压不住了, 深层参数的一次优化步长就会过大, 导致一直无法进入极小值位置(在外侧兜圈), 产生**欠拟合(模型训练不到位)**问题; 第二, 更坏, 数值大到一定程度, 会超出计算机浮点真值上限, 这一般会被标识成 NaN(Not a Number), 被计算机拒绝处理, 参数一直不更新.

这块内容值得讲的有很多. 但是限于篇幅和课程性质, 无法详细展开. 希望大家可以自己课后查一查相关的东西.

[这里推荐一些东西]

#### 梯度下降的问题与解决方法

大家可以看出, 梯度下降的这种下降其实是盲目的, 很容易下降到局部极小而不是全局最小. 为了防止梯度下降法优化的模型参数陷入局部极小, 我们引入了**正则化方法**.

正则化方法其实本质是为了预防**过拟合**问题. 当数据集过小(或者说是种类过于单一), 无法比较好地代表数据的真实情况时, 对于神经网络这种表示能力强的模型来说, 很容易遇到过拟合的问题. 其表现为在训练数据集上效果非常好, 但是在真实数据上却差很多. 直观理解是模型学习到了不属于总体的知识. 例如你的数据集中大部分是中国的气象数据, 其它地方的数据很少. 然而真实场景中是在全球范围内随机抽选地区进行气象预测. 中国的气象数据中肯定包含气象变化的一般规律(我们希望模型学到的), 但是也包含了中国这个特定地区的特有变化(其实我也们希望模型学到, 但是要注意区分地区), 如果数据集中大多数样本来自中国, 模型就会认为中国地区的特色气象就是全世界气象的共性. 导致从整体来看性能下降.

在神经网络训练中, 传统的正则化方法有 **L1 正则化**与 **L2 正则化**. 不过现在更多使用 **Dropout 方法**. Dropout 效果非常好.

这种从某个子集中学习到的成果, 运用到父集合中的过程, 我们称之为**泛化**.

## AI 技术栈

![](static/SnRfblBBCoKjlxxFCUjcAtbRnwb.png)

### AI 应用

大家可能在很多的地方听说过 AI 的应用. 其中最让人感觉像 AI 的应用就是各种 AI 聊天应用(ChatGPT, Gemini, Claude, 千问, 豆包等). 正所谓语言是思想的载体, 通过语言, 人们第一次能直观地感受到 AI 的"思想".

AI 的应用远远不止是这类 ChatBot 以及其衍生出的服务(代码补全, agent 等). 还有图像生成, 推荐算法, 自动驾驶等. 而现代的 AI 应用往往核心是深度学习. 其参数非常巨大, 往往需要在集群上进行分布式训练, 这导致代码比较复杂. 于是有人开发出了训练/推理框架.

### 推理/训练框架

推理框架(常见的是 vLLM 与 TensorRT), 是为了便于模型部署而被开发出的框架. 它们往往专门为了部署场景进行了大量优化, 以达到强悍的性能.

训练框架(Megatron-LM 与 DeepSpeed), 是为了便于大规模模型训练而被开发出的框架.

### 深度学习框架

一般来说, 虽然推理与训练框架有大量自己的优化, 其核心依然会大量使用深度学习框架. 深度学习框架提供了自动微分功能和方便的高性能张量计算 API. 有关深度学习框架的使用与实现原理, 我们将在深度学习框架的章节进行更多介绍.

### 算子库与驱动

本课程基本不涉及这一层级的内容. 算子库是把一些通用操作(例如加减乘除, 矩阵乘, 求和等)的代码提前编写好, 并且进行了极致的优化. 深度学习框架一般会调用算子库中的算子来完成计算.

驱动更加底层, 其涉及更多架构知识, 是软件与硬件的交接层, 这里不作介绍.

### 硬件

所有的计算最终都由硬件完成. CPU 设计出来是为了应对更加广泛的场景, 而不是大量的张量计算(线性代数计算), 导致其许多为了其它情况设计的电路用处不大. 相同的面积下, 如果我们把更多的面积用于计算单元, 而不是某些情况下的加速, 就可以达到更好的性能. 这就是垂直领域硬件设计.

相比之下, GPU 比 CPU 更擅长处理线性代数运算. 线性代数中存在大量的并行性, 依赖关系一般不复杂. 因此 GPU 的控制粒度可以较粗, 控制电路可以较简单. GPU 设计初衷是计算机图像处理, 本身就是为了这种高并行, 低控制场景而设计.

直观来看, GPU 相比 CPU, 有更多的计算核心, 可以同时进行规模更大的运算, 因此其在线性代数场景下的计算能力更强. 更多介绍同学可以自己查阅资料.

# 第二章 Python 基础

Python 已经是一门非常有名的语言. 相信不少同学都听说过, 甚至有同学已经熟练掌握.

很多人认为 Python 最大的优势是**简洁**, 可读性强. 阅读良好的 Python 程序如同阅读英语一般. 诚然, Python 在设计上就在追求一种简洁与和谐. 但其实 Python 对于普通开发者来说最大的优点是**强大的生态**. 在深度学习领域, Python 是几乎所有框架的编程 API 第一选择.

就算你不做一个 AI 开发者, 你也最好学一学 Python. 这门语言本身的学习成本可以说是最低的, 而扩展又十分强大. 你需要快速搭建一个 web 应用时, 你不需要学习前端框架与 Spring, 使用 fastapi, streamlit 这些 python 库就可以迅速搭建一个基本的 web 应用. 可以这么说, Python 是最适合非程序员的语言, ~~也是程序员最喜欢的玩具.~~

## Python 的历史

**Python** 是在 1980 年代后期所构思出来的编程语言, 并于 1989 年 12 月由荷兰 CWI 的[吉多·范罗苏姆](https://zh.wikipedia.org/wiki/%E5%90%89%E5%A4%9A%C2%B7%E8%8C%83%E7%BD%97%E8%8B%8F%E5%A7%86)开始进行编程发展.

Python 源代码遵循 GPL(GNU General Public License)协议.

Python 2.0 版于 2000 年 10 月 16 日发布, 引入垃圾回收器.

Python 3.0 版是一个主要的"向后不兼容"(backwards-incompatible)版本, 经过长时间的测试之后, 于 2008 年 12 月 3 日发布.

现在我们所说的 Python, 大多指的是 Python3.

## Python 原理直观解释

大家在程序设计课上学过, C++ 是通过编译器, 将源代码文件编译成可执行文件, 然后才能在机器上执行.

而 Python 并不使用这种方法. 它使用"解释器"来运行源代码. 这个解释器有多个实现, 例如 CPython, PyPy, Numba. 不过我们一般默认是 CPython. 当你写好 Python 源代码文件后, 使用 Python 解释器执行这个文件. 解释器会将程序"一行一行"地实时翻译成机器指令, 来使其在机器上运行.

> 为了方便理解, 这里并不严谨. 实际上解释器运行程序的机制较为复杂, 并不是一行一行. 而是被转换成"字节码", 类似机器指令, 随后解释器会像 CPU 执行指令一样处理字节码. 这其中, 会使用类似"指令预取"等与 CPU 类似的技术.
> 而 PyPy 与 Numba 的解释器实现, 使用名为 JIT(Just-in-Time Compile, 即时编译)的技术, 与 CPython 的类似 CPU 的字节码指令与"求值循环"有所区别.

因此, 运行 Python 不需要经过耗时的编译流程, 只需要等待解释器的启动, 程序就可以开始运行.

## 安装与环境配置

在安装 Python 之前, 先要说明:

- 你们之后实际上很少用全局环境的 Python, 也就是你待会要安装的 Python. 本课程后面会讲解 Python 虚拟环境相关的内容. 在那之后实际上大家就使用 conda 与 uv 创建的虚拟环境中的 Python 了.
- 有关环境配置中的 VsCode 相关内容. 这里假定你们已经安装了 Visual Studio Code. 注意是 Visual Studio Code, 而不是 Visual Studio. 后者就是在程序设计课上要求安装的.

### Windows

不要从什么应用商店下!!!

前往 Python 官网(python.org), 找到 Downloads 页面:

![](static/UTMUbLLNIoQJ8exXsSCcXdWonRd.png)

然后直接点击"Download Python install manager":

![](static/RoYEbCGQboDY7hxwwrqcHZCenpf.png)

国内网络环境下载会比较慢...

[施工, 加速站点]

下载好之后, 双击运行该文件. 在弹出的窗口中点击确认.

[施工]

### MacOS

#### 安装 Homebrew（Mac 核心包管理器）

为了绕过网络环境限制，使用了国内维护的安装脚本，并选择了更稳定的镜像源。

- **执行安装脚本**：
- `/bin/bash -c "$(curl -fsSL ``https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)``"`
- 跟着这里的脚本的提示一步一步来就行了
- **关键配置选择**：

  - **本体下载源**：选择 `2` (Gitee)，避开了 `raw.githubusercontent.com` 的连接报错。
  - **软件镜像源**：选择 `5` (阿里巴巴)，确保后续安装软件的速度。
- **激活环境变量**：
- 执行 `source ~/.zprofile` 将 `brew` 命令添加到系统搜索路径。
- **验证安装**：
- `brew -v`。

![](static/Gv1mbDGwYoeK12xRp3gcdCeCnbg.png)

![](static/MyMDbJvm7oIkVZxmmWac2YmMnyb.png)

![](static/AGd7bs4yNoO39HxGTnvcD39inLE.png)

#### 全局语言层：Python 3.10（我这边选择的是一个比较稳定的 python 版本）

虽然 macOS 自带 Python，但为了稳定性和后续开发，建议通过 Homebrew 安装一个独立的全局版本。

##### 2.1 安装命令

Bash

```
brew install python@3.10
```

---

##### 2.2 路径说明

- **安装位置**：`/opt/homebrew/bin/python3`（可以用 which python3.10 来看在哪）
- **用途**：用于运行简单的 Python 脚本、安装通用的工具包。

#### 实验室管理层：Miniconda（比较方便一点）

对于 AI、机器学习或复杂项目，Miniconda 是实现“环境隔离”的核心工具。

##### 3.1 安装 Miniconda

Bash

```
brew install --cask miniconda
```

##### 3.2 初始化与激活

安装后需初始化 Shell，使 `conda` 命令生效：

Bash

```
/opt/homebrew/bin/conda init zsh
source ~/.zshrc
```

_现象_：终端提示符前出现 `(base)`，代表进入了 Conda 基础环境。

##### 3.3 配置国内镜像源 (Conda)

为了加速第三方库的下载，配置清华镜像源：

Bash

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

---

#### 编辑器配置：VS Code

去 [VS Code 官网](https://code.visualstudio.com/)。正常下载 dmg，然后安装就可以了。

![](static/Etbkb7YngoWI3NxJDSJcNatqnPh.png)

---

#### 如何开始写代码？（此部分都是跟 windows 那边通用的）

##### 步骤 A：创建独立虚拟环境（这部分建议自己学习一下 conda 环境相关的命令，网上很容易查得到，建议动手试一下）

永远不要在 `base` 环境下装库。为新项目创建一个新环境：

Bash

```
conda create -n my_project python=3.10
conda activate my_project
```

##### 步骤 B：在 VS Code 中关联环境

1. 在编辑器中打开项目文件夹。
2. 按 `Cmd + Shift + P` -> 输入 `Python: Select Interpreter`。
3. **选择你的 Conda 环境**（通常会标注为 `'my_project': conda`）。

---

#### 维护常用命令速查

---

**配置提示**：

- **M 系列芯片加速**：安装 PyTorch 后，代码中可以使用 `device = torch.device("mps")` 来调用 Mac 的 GPU 进行加速。
- **保持整洁**：所有的 Conda 环境现在都规范地保存在 `/opt/homebrew/Caskroom/miniconda` 目录下，不再污染系统路径。

### Linux

## Python 语法入门

### Hello World

在终端直接输入 `python` 进入终端交互模式:

```bash
PS C:\Users\MSforAI> python
Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

>>> 是输入提示符. 在这后面输入 print("Hello, world!"):
>>>
>>

```bash
Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> print("Hello, world!")
Hello, world!
>>>
```

> 退出交互模式: 输入 `exit()`. 即调用退出函数.

不过 Python 肯定不止有这种在终端里输入一行, 解释一行的模式. Python 更常用的是使用脚本.

创建一个后缀为 `.py` 的文件(Python 源文件的后缀是 `.py`), 内容为:

```python
print("Hello, world!")
```

> Python 不需要什么 main 函数作为程序入口. 程序默认从第一行开始执行. Python 语句之间使用换行符\n 作为分隔, 而不使用分号;进行分隔.

在当前目录下的终端中输入 `python xxx.py`(xxx 是你的文件名字), 就可以运行.

```bash
PS C:\Users\MSforAI\Desktop> python main.py
Hello, world!
```

### 变量与运算

#### 变量赋值

Python 中变量采用等于号 `=` 进行赋值, 并且**不使用**类型声明:

```python
a = 123
var = 0.114514
b = "Hello, MS for AI!"
```

这是因为 Python 是一门"动态类型"语言.

> 动态类型语言: 程序在运行时才去检查类型. 例如:
>
> ```python
> ```

a = 1
b = "2"
c = a + b

```
> 这个Python程序会报错. 因为`a`与`b`一个是整数类型, 一个是字符串类型, 不能执行`+`操作. Python在运行程序后, 会在内部隐式自动标注变量的类型, 只有运算的时候出现问题了, 才会报错.

Python中能实现随便修改类型的操作:

```python
var = "114514"
var = 1919810
```

这个操作是允许的. 程序结束前, `var` 的值最后是 1919810.

Python 中支持多个变量同时赋值, 例如:

```python
a = b = 123
a, b, c = 1, 1.2, "3"
```

#### 变量基本类型

Python 中的变量有这些标准类型:

- 数字 Numbers
- 字符串 String
- 列表 List
- 元组 Tuple
- 字典 Dictionary
- 布尔值 Bool

##### 数字

数字用于存储数. 它们也有以下四种类型:

- `int` 有符号整数
- `long` 长整型，也可以代表八进制和十六进制**(python3 后不存在 long, 与 int 类型合并)**
- `float` 浮点数
- `complex` 复数

```python
a = 1    # int
# python3 中不存在long类型
b = 0.1    # float
c = complex(1, 2)    # complex
c = 1 + 2j    # complex
c = 1 + 2J    # complex
```

> #是 Python 的注释标识符. 从#开始直到换行会被视为注释内容, 不会被 Python 解释器运行.

##### 字符串

字符串可以包含大多数字符:

```python
s = "1a_-%$@MSfor AI! 你好, 世界!\n\t"
```

你也可以定义空字符串:

```python
s = ""
```

字符串也可以用单引号定义. 与双引号没区别. 这样可以方便地在字符串中内含":

```python
s = 'abc'
s = '她说: "你是好人, 但是...你没有学过MS for AI. 抱歉!"'
```

定义很大的字符串, 可以使用 3 个双引号:

```python
s = """Missing Semester for AI
copyright@131AIClub
人工智能教育中缺失的一课
这是一个很大大大大大大大的字符串...
没有了
"""
```

> 注意, """也可以用于大规模注释:
>
> ```python
> ```

"""
这是一个很大大大的注释
一般会用于编写文档.
"""

"""这也是注释, 换不换行并不会影响"""

```

字符串的一个很重要的操作是**切片**. 切片使用中括号来标记一个下标区间, 左闭右开:

```python
"""
特别要注意: 
    1. 索引从0开始.
    2. 切片的右边值无法取到. (左闭右开)
"""
s = 'Missing Semester for AI'
print(s[0: 7])    # Missing
print(s[8: 16])    # Semester
print(s[0: 2])    # Mi
```

字符串切片还可以有第三个值, 含义是"步长", 即取数每一步的长度. 默认为 1, 即区间内全取. 步长也可以取负数.

```python
s = 'Missing Semester for AI'
print(s[0: 7: 2])    # Msig
print(s[8: 16: 3])    # See
```

切片的三个值都是可以不输入的. 默认值分别为:

- 第一个值(起始位置): 0, 即串的开头.
- 第二个值(结束位置): len(s), 即串的长度.
- 第三个值(步长): 1, 即全取.

> `len` 函数: 返回串的长度:
>
> ```python
> ```

s = "012345"
length = len(s)    # 6

```

例如:

```python
s = 'Missing Semester for AI'
print(s[::])    # Missing Semester for AI
print(s[:: -1])    # IA rof retsemeS gnissiM
print(s[2:])    # ssing Semester for AI
print(s[: 3:])    # Mis
```

##### 列表

List(列表) 是 Python 中使用最频繁的数据类型.

列表可以完成大多数集合类的数据结构实现. 它支持数字, 字符串甚至可以包含列表(即嵌套).

准确来说, 列表几乎可以装任何对象. 并且不要求列表元素是同一个类型.

```python
l = [1, "2", [3, 4]]
l = []    # 空列表
l = list()    # 这个也是空列表
```

列表是动态长度的, 可以通过 `append` 方法来添加新元素:

```python
l = []
l.append("I")
l.append("love")
l.append(['MSforAI', '!'])
print(l)    # ['I', 'love', ['MSforAI', '!']]
```

列表删除元素可以使用 `pop` 和 `remove` 方法.

- `pop` 方法接收一个参数作为要删除的元素的索引. 缺省值为-1, 即最后一个元素. `pop` 方法会返回删除的元素:

```python
l = [1, 2, 3]
pop_value = l.pop()
print(l)    # [1, 2]
print(pop_value)    # 3

l = [1, 2, 3]
pop_value = l.pop(1)
print(l)    # [1, 3]
print(pop_value)    # 2
```

- `remove` 方法接收一个匹配值, 会删除第一个匹配到的元素. `remove` 没有返回值(返回 `None`):

```python
l = [1, 2, 'MSforAI', '天天天国地狱国']
remove_value = l.remove('MSforAI')
print(remove_value)    # None
print(l)    # [1, 2, '天天天国地狱国']
```

列表可以使用索引访问和修改元素:

```python
l = [1, 2, 3]
print(l[1])    # 2
l[0] = 'new'
print(l)    # ['new', 2, 3]
```

列表支持切片操作. 逻辑与字符串基本相同:

```python
l = [0, 1, 2, 3, 4, 5]
s = l[::-1]    # [5, 4, 3, 2, 1, 0]
s = l[:2]    # [0, 1]
```

##### 元组

在当前阶段, 你可以认为元组与列表最大的区别就是: 元组是不可变的.

定义元组使用括号():

```python
t = (1, 'it', 'is a tuple!')
t = (,)    # 空元组
t = tuple()    # 空元组
```

> 为什么 `t=(,)` 要加逗号?
> 这是因为, 如果你写()会出现歧义: 为什么()不是个空表达式? 虽然 Python 解释器确实会认为()是个空元组, 但是为了避免歧义, 最好写个逗号.
> 实际上, 你在定义只有一个元素的元组时, 必须要写个逗号:
>
> ```python
> ```

t = (1,)

```
> 如果你不写逗号, Python解释器会认为这是个表达式, 最终`t`的类型是`int`
> ```python
t = (1)
print(type(t))    # <class 'int'>
```

你可以通过索引来读取元组中的值, 但是你不能修改(准确来说, 是你不能更换元组中的对象):

```python
t = (1, 2, 3)
print(t[2])    # 3
t[0] = 123    # 这行会报错

t = (1, 2, [1, 2])
t[2].append(3)    # 可以!
print(t)
```

这里列表可以被更新是因为, 你没有更换元组中的对象, 列表还是那个列表, 只是列表自己的内含值变了.

由于元组不能修改, 所以元组是定长的.

元组也支持切片操作, 逻辑类似字符串:

```python
t = (1, 2, 3, 4, 5)
print(t[:3])    # (1, 2, 3)
```

##### 字典

字典是无序的对象集合. 由键值对组成. 你可以通过键 key 来快速查找到值 value.

> 字典在实现上使用哈希表(散列表).

字典使用花括号{}定义, 使用引号:来分隔键值对:

```python
d = {1: '1', 2: '4', 3: 'AI', 4: 'vibe'}
d = {}    # 空字典
d = dict()    # 空字典
```

字典的值可以是绝大多数的 Python 对象. 但是字典的键必须是**可哈希的**. 列表与元组是不可哈希的, 因此不能作为字典的键.

> Python 判断是否可哈希, 是通过该对象是否实现 `__hash__` 魔术方法来判断的. 这部分的内容, 有兴趣的同学可以自己查询资料来学习. 在后续学习了面向对象后, 大家对这一块会有更清楚的认识.

```python
d = {
    (1, 2, 3): 654    # 报错!
}
```

字典可以通过键(key)来查找值(value):

```python
d = {
    1: "i",
    2: "love",
    "3": 131,
    114514: ['c', 'l', 'u', 'b'],    # 最后一个键值对的逗号,可省略
}
print(d[1])    # i
print(d['3'])    # love
print(d[114514])    # ['c', 'l', 'u', 'b']
```

字典添加键值对最常用以下两种方式:

- 直接赋值
- `update` 方法

直接赋值的形式如下:

```python
d = {}
d['key1'] = 'value1'
d['key2'] = 'value2'
d[131] = ('ai', 'club')
print(d)    # {'key1': 'value1', 'key2': 'value2', 131: ('ai', 'club')}
```

`update` 方法形式如下:

```python
d = {}

# 第一种update使用方法
d.update({1: "131", '2': 666})    
d.update({5: '222'})

# 也可以使用参数式定义, 参数名会被转为string类型键
d.update(club=131)    
d.update(roxy='wife', msforai='love')
print(d)    # {1: '131', '2': 666, 5: '222', 'club': 131, 'roxy': 'wife', 'msforai': 'love'}
```

删除字典的键, 可以使用 `del` 语句, 或者 `pop` 方法.

更常用的是 `pop` 方法, `pop` 方法会返回删除值:

```python
d = {'club': 131, 'ms': ['y', 'e', 's'], 'key': 100}
value = d.pop('ms')
print(value)    # ['y', 'e', 's']
print(d)    # {'club': 131, 'key': 100}
```

`del` 语句的形式:

```python
d = {'club': 131, 'ms': ['y', 'e', 's'], 'key': 100}
del d['ms']
print(d)    # {'club': 131, 'key': 100}
```

> `del` 语句还有更多的用处. 这里限于课程性质, 不介绍. 想要学习的同学可以查找相关资料来学习.
> C++ 中 delete 语句很重要, 而在 Python 中 `del` 语句却没那么重要. 这是因为 Python 中有**垃圾回收机制**.** **Python 解释器维护了一个垃圾收集器(Garbage Collector, GC), 它会按照一个规则(通常是引用计数)来回收用户不使用的对象. 因此 Python 程序员一般情况下不需要考虑回收问题.

##### 布尔值

Python 中的布尔值类似数字型变量. 其只有两种值: `True` 和 `False`:

```python
# 布尔值要大写: 是True和False, 不是true和false.
t = True
f = False
```

#### 基本运算

##### 数字型变量加减乘除

一些大家肯定懂的东西, 这里就快速介绍.

数字型之间的加减乘除是很符合直觉的:

```python
"""
int与float之间运算, 不需要显式类型转换, 会默认全转成float.
"""
# 加
a = 1
b = 0.1
print(a + b)    # 1.1

# 减
a = 1
b = 3.1
print(a - b)    # -2.1

# 乘
a = 1.1
b = 2
print(a * b)    # 2.2

# 除
a = 1
b = 3
print(a / b)    # 0.3333333333333333

# 整除
a = 7
b = 3
print(a // b)    # 2

# 取余(模)
a = 7
b = 3
print(a % b)    # 1
```

复合运算符, 即 `+=`, `-=` 等, Python 也是支持的:

```python
a = 1
a += 1
print(a)    # 2
a -= 5
print(a)    # -3
a /= 2
print(a)    # -1.5
...    # 你没看错, ...真的是个Python关键字, 与pass差不多
```

接下来介绍一些特殊的运算:

##### 字符串加法与乘法

字符串的加法实际就是拼接:

```python
s1 = 'Hello, '
s2 = 'world!'
print(s1 + s2)    # Hello, world!

s = '131' + 'AI' + 'Club'
print(s)    # 131AIClub
```

字符串乘法就是重复拼接:

```python
s = '131' * 3
print(s)    # 131131131
```

##### 列表加法与乘法

列表加法与乘法与字符串类似:

```python
l1 = [1, 2, 3]
l2 = ['1', '2', '3']
l = l1 + l2
print(l)    # (1, 2, 3, '1', '2', '3')

l = [1, 3, 1] * 3
print(l)    # (1, 3, 1, 1, 3, 1, 1, 3, 1)
```

##### 元组的加法与乘法

与列表, 字符串基本相同:

```python
t1 = (1, 2, 3)
t2 = ('1', '2', '3')
t = t1 + t2
print(t)    # [1, 2, 3, '1', '2', '3']

t = [1, 3, 1] * 3
print(t)    # [1, 3, 1, 1, 3, 1, 1, 3, 1]
```

##### 字典的合并运算 |

字典的合并运算 `|` 是在 Python3.9 引入的. 它可以合并两个字典:

```python
d1 = {1: '1', 2: '2'}
d2 = {'1': 1, '2': 2}
d = d1 | d2
print(d)    # {1: '1', 2: '2', '1': 1, '2': 2}
```

### 分支与循环

程序的基本控制结构就是顺序, 分支, 循环. Python 作为健全的语言, 肯定是可以实现这些基本的控制结构.

#### 分支

##### if-else

Python 实现分支需要使用 `if` 与 `else` 关键字:

```python
if True:
    a = 1
else:
    a = 2
print(a)    # 1
```

注意程序缩进: 分支的语句体需要向内缩进一个单位(一般是 4 个空格)

多分支可以使用 `else if`:

```python
flag = 5
if flag > 10:
    print('flag > 10')
else if flag > 5:
    print('flag > 5')
else if flag > 0:
    print('flag > 0')
else:
    print('flag <= 0')
# flag > 0
```

注意, 在多分支结构中, 遇到了一个匹配项, 则后续全部分支都会跳过.

##### match-case

Python3.10 引入的新语法, 对标 switch 语法.

```python
status = 3
match status:
    case 1:
        print('情况1')
    case 2:
        print('情况2')
    case 3:
        print('情况3')
    case _:
        print('未知情况')
```

留个印象就可以了, 很少用.

#### 循环

循环主要由两种关键字实现, 分别是 `while` 和 `for`.

##### While

`while` 关键字后面需要写一个布尔值表达式, 代表循环继续条件. 循环会一直进行到条件不满足为止:

```python
i = 0
while i < 10:
    print(i)
    i += 1

# 这是一个死循环
while True:
    pass
```

如果想要中途直接退出循环(而不是等待循环条件不满足), 可以使用 `break` 语句:

```python
# 这个循环只会输出到6
i = 0
while i < 10:
    print(i)
    if i > 5:
        break
    i += 1
```

还有一个语句是 `continue`. 它会直接开始下一次循环, 跳过 `continue` 后面执行的语句:

```python
i = 0
while i < 131:
    print(i)
    continue    # 这里会导致死循环!
    i += 1
```

`while-else` 语句. 实际上 `while` 还有一个 `else` 语句, 不过大多数情况下我们是不使用的. 它的含义就是在 `while` 循环结束后, 执行 else 语句中的内容. 但是注意: 通过 `break` 退出的循环, 不会执行 `else` 的内容.

```python
i = 0
while i < 10:
    print(i)
    i += 1
else:
    print('loop1 finish!')

i = 0
while i < 10:
    print(i)
    if i > 5:
        break
    i += 1
else:
    print("loop2 finish!")
# 你只会看到loop1 finish!
```

##### For

`for` 比 `while` 用的会更多一些~~(我的身边统计学)~~. `for` 循环需要指定循环变量与迭代器. `range()` 函数会生成一个迭代器. 它接收整数作为输入, 返回一个按照一定规则迭代整数的迭代器. 列表与元组也是可迭代的, 会依次按顺序迭代出其内含的元素.

```python
for i in range(10):
    print(i)

l = [131, 'ai', 'club', 'ms', 'for', 'ai']
for item in l:
    print(item)

t = ('it', 'is', 'a', 'tuple')
for item in t:
    print(item)
```

循环变量有时候我们是不关心的, 即在循环中我们不会使用. 例如说, 我们只是希望一个相同的过程执行 100 次, 但是我们并不需要知道我们执行到第几次. 而且, 最关键的是, 我们懒得想变量名. 这个时候, 你可以用占位符来替代变量标识符:

```python
for _ in range(100):
    print("I don't care i.")
```

除了替换循环变量标识符, 占位符也可以替换变量赋值的位置:

```python
t = ('131AIClub', 'MSforAI', 114514)
name, coure, _ = t    # 我不关心最后一个值, 而且我懒得想变量名
```

在终端交互环境中, 占位符 `_` 可以用于指代最近的一个表达式的值:

```bash
Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> 12.44*5 + 0.22
62.419999999999995
>>> a = 10
>>> _
62.419999999999995
>>>
```

有关迭代器的更多讲解, 介于课程性质, 不介绍. 感兴趣的同学可以自己查阅资料学习.

### 函数

Python 使用 `def` 关键字声明函数. 分别需要声明: 函数标识符(函数名), 函数参数(arguments). 并使用 `return` 语句来声明返回值. 如果没有声明返回值, 则返回 `None`.

```python
def func(x, y):
    z = x + y
    return z

def foo():
    pass
```

通过函数标识符 +()来调用函数:

```python
def func(x, y):
    return x + y

print(func(1, 2))    # 3
print(func(x=1, 2))    # 报错! 关键字指定要在最后
print(func(1, y=2))    # 3
print(func(x=1, y=2))    # 3
```

调用函数时, 不写参数名的是**位置参数**, 写参数名的是**关键字参数**.

#### 递归

Python 是支持递归的, 例如下面这个计算 Fibonacci 数的程序:

```python
def fibonacci(n: int) -> int:
    if n <= 2:    return 1
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(1, 11):
    print(fibonacci(i), end=' ')
```

> Python 类型注释. Python 虽然不需要声明变量类型, 但是你也可以通过类型注释来声明变量类型. 类型注释在运行时不会有任何影响, 但是在编写程序时的 lsp 分析中很有用: lsp 可以进行类型推导, 提前发现一些类型错误.
> 类型注释的基本方法是在变量的后面加 `:<类型>`, 例如 `x: int`, `s: str`. 函数返回值的类型注释可以通过在函数声明行添加 `-><类型>` 来实现, 例如 `def f() -> None:`, `def g(x: int) -> str:`

#### 闭包

闭包. Python 函数的参数与返回值可以是函数!

```python
"""
这是一种叫做"装饰器"的Python编程技巧.
@是一个语法糖:
例如@func1, 就是把后面第一个定义的函数, 作为参数传入func1中, 并使用func1的返回值替换这个定义的函数.
"""
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec('Params')
R = TypeVar('Return')

def dec(f: Callable[P, R]) -> Callable[P, R]:
    def g(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f.__name__ + " 执行!")
        return f(*args, **kwargs)
    return g

@dec
def add(x: int, y: int) -> int:
    return x + y

print(add(1, 2))
```

> 可变参数 `*args` 与 `**kwargs`: `*args` 用于接收任意数量的位置参数，这些参数会被收集到一个**元组**中. 而 `**kwargs` 用于接收任意数量的关键字参数, 会被收集进一个**字典**中.

闭包的实际定义其实比较复杂, 并不是简单的"参数与返回值可以是函数". 这一块内容其实不能算是 Python 入门了...

大家只要有这个思想就可以了, 知道 Python 中函数是一个比较灵活的东西, 实际上只是一个 `Callable`, 可调用的东西.

### 面向对象

Python 是一门面向对象的语言. 尽管一直有人在抨击 OOP(面向对象英文简写)模式, 但是面向对象确实是利于人类程序员建模现实问题, 组织程序结构. 不管你有没有对象, 你都得先想办法面向对象. 每当我写一个新项目时, 第一个写的总是继承抽象类的类, 不管有没有必要写这个类. 但是, 最佳实践这一块 hhh.

简单回顾一下面向对象的一些基本概念. 类是对象的蓝图或模板, 对象是类的实例. 类定义了对象的属性和行为. 我们把一些事物分类进一个类中, 并且抽象出一些共同属性, 就构成了类.

例如说, 我想要编写一个服务器后端, 它的功能是接收用户请求, 返回用户需要的图片(一个很常见的例子是图床, 虽然那个是随机的). 我其实可以选择为用户请求实现一个类, 每当接收一个用户请求, 我就实例化一个用户请求对象. 而用户请求类内部包含请求的用户名, 请求的图片 id, 请求的优先等级等属性, 这些属性是共通的. 如果我不适用类来实现, 那这将变得很麻烦, 我不得不维护很多的变量, 没有办法把这些属性组织起来.

Python 中定义类使用 `class` 关键字:

```python
class A:
    ...

a = A()    # 实例化一个对象
```

不过现在这个类啥都没实现...

一般来说首先要实现的是构造函数. Python 中规定了类的构造函数为 `__init__` 魔术方法:

```python
class Car:
    def __init__(self, color: str) -> None:
        self.color = color
    
    def print_color(self) -> None:
        print(f'My color is {self.color}!')

car = Car('red')
car.print_color()
print(car.color)
```

`self` 不需要外部提供, 代表对象本身. 这个参数是不能放在中间, 只能放在第一个的. 如果没有 `self` 参数, 该方法被称为静态方法(static method). 静态方法需要 `staticmethod` 装饰器才能实现.

#### 继承与多态

Python 类当然是支持继承与多态的.

类的继承形式如下:

```python
class Animal:
    def __init__(self, name: str) -> None:
        self.name = name
    
    def move(self) -> str:
        return f'{self.name} is moving!'
    
    def speak(self) -> str:
        return "Some sound..."

class Dog(Animal):
    def speak(self):
        return f'{self.name} says Woooof!'

dog = Dog('puppy')
print(dog.speak())
```

类可以多层继承, 也可以多重继承:

```python
class Machine:
    def work(self) -> str:
        return 'Machine is working...'

class Vehicle(Machine):
    def work(self) -> str:
        return 'Vehicle is working...'

class Car(Vehicle):    # 多层继承
    def work(self) -> str:
        return 'Car is working...'

class Flyable:
    def fly(self) -> str:
        return 'Something is flying...'

class Plane(Vehicle, Flyable):    # 多重继承
    def fly(self) -> str:
        return 'Plane is flying...'

car = Car()
print(car.work())
plane = Plane()
print(plane.fly())
print(plane.work())
```

Python 的类默认继承 `object`:

```python
class A(object):
    ...

class A:    # 等价
    ...
```

如果你想在子类中扩展而非完全替换父类的方法, 你可以使用 `super()` 函数. `super()` 将返回一个类似父类对象的东西:

```python
class Animal(object):
    def __init__(self, name: str) -> None:
        self.name = name

class Dog(Animal):
    def __init__(self, name: str, age: int) -> None:
        super().__init__(name)
        self.age = age

dog = Dog('dog_name', 114514)
print(dog.name, dog.age)
```

多态即同一个接口由多个不同类型使用, 实现不同的功能. 例如:

```python
import math

class Shape:
    def area(self) -> float:
        raise NotImplementedError()

class Triangle(Shape):
    def __init__(self, a: float, b: float, c: float) -> None:
        self.a, self.b, self.c = a, b, c
    
    def area(self) -> float:
        a, b, c = self.a, self.b, self.c
        p: float = 0.5 * (a + b + c)
        return math.sqrt(p * (p - a) * (p - b) * (p - c))

class Circle(Shape):
    def __init__(self, r: float) -> None:
        self.r = r
    
    def area(self) -> float:
        r = self.r
        return math.pi * r ** 2

class Square(Shape):
    def __init__(self, h: float, w: float) -> None:
        self.h, self.w = h, w
    
    def area(self) -> float:
        h, w = self.h, self.w
        return h * w
    
triangle = Triangle(3, 4, 5)
circle = Circle(1.5)
square = Square(2, 4)
print(triangle.area())
print(circle.area())
print(square.area())
```

这里都继承了 `area` 接口, 但每个子类有自己的实现. 以此来实现多态.

#### 类型检查

Python 可以使用 `isinstance` 与 `issubclass` 进行类型检查:

```python
x: int = 1
print(isinstance(x, int))    # True

class A:    pass
a: A = A()
print(isinstance(a, A))    # True

y: float = 0.
print(isinstance(y, int))    # False
print(isinstance(y, (int, float)))    # True, 多类型判断
```

`issubclass` 用于检查一个类是否属于某个类的子类:

```python
class Animal:    pass
class Dog(Animal):    pass

print(issubclass(Animal, Dog))    # False
print(issubclass(Dog, Animal))    # True
```

## Python 包管理

> ~~技术的本质就是调包~~~~! ~~

「复用」是软件工程中一个十分重要的概念。如果我们想要实现的功能在项目的其他地方已经存在，那么我们就不必再次实现它。我们可以将程序的各个部分共享的逻辑提取出来，这样一来既可以避免代码的冗余，修改时也可以仅在一处修改，而不是在所有相同逻辑出现的地方进行修改。许多编程语言为了支持这样的复用都发展出了相应的机制，例如函数允许实现对于过程的复用，而语言中的类和对象和相关的 OOP 机制允许实现更加复杂的，对于数据结构和过程的复用。

然而，以上所说的这些复用机制仅仅局限于单个项目内。实际开发中，我们经常需要跨项目或跨团队共享和复用已有的功能。这就引出了「跨项目复用」的概念。我们可以把共同的逻辑提取出来，让这些逻辑本身成为一个单独的项目进行维护。在今天，你已经知道，这样的项目叫做「库」。今天的数字基础设施正是由一个又一个或大或小的「库」组成。得益于计算机科学的发展和自由软件运动，我们在今天可以轻而易举地获取到别人发布的开源库，并用在自己的项目中。这样一来，复用的规模被再一次扩大了，它并非局限于某个组织、国家或地区，而是所有人类程序员之间的复用。~~(为什么一定要是人类?)~~

![](static/ESBAbbAr5o4mn1xFWq0c2JzWnuh.png)

[自行查阅 python 中的模块与包]

在前面的课程中，你已经使用过一些库了。它们通过 `import xxx` 的形式被导入，在那之后你就可以使用他们。但是到此为止，你使用的都是 python 自带的[标准库](https://docs.python.org/zh-cn/3/library/index.html)。想要使用其他的库，你需要使用**包管理器**。python 官方提供了一个称为 **pip** 的包管理器，它随着 python 解释器的本体安装。想要使用它，只需要运行 `pip`：

```
$ pip

Usage:
  pip <command> [options]

Commands:
  install                     Install packages.
  download                    Download packages.
  uninstall                   Uninstall packages.
  freeze                      Output installed packages in requirements format.
  list                        List installed packages.
  show                        Show information about installed packages.
  check                       Verify installed packages have compatible dependencies.
  config                      Manage local and global configuration.
  search                      Search PyPI for packages.
  cache                       Inspect and manage pip's wheel cache.
  index                       Inspect information available from package indexes.
  wheel                       Build wheels from your requirements.
  hash                        Compute hashes of package archives.
  completion                  A helper command used for command completion.
  debug                       Show information useful for debugging.
  help                        Show help for commands.

General Options:
  -h, --help                  Show help.
  --isolated                  Run pip in an isolated mode, ignoring environment variables and user configuration.
  -v, --verbose               Give more output. Option is additive, and can be used up to 3 times.
  -V, --version               Show version and exit.
  -q, --quiet                 Give less output. Option is additive, and can be used up to 3 times (corresponding to
                              WARNING, ERROR, and CRITICAL logging levels).
  --log <path>                Path to a verbose appending log.
  --no-input                  Disable prompting for input.
  --proxy <proxy>             Specify a proxy in the form [user:passwd@]proxy.server:port.
  --retries <retries>         Maximum number of retries each connection should attempt (default 5 times).
  --timeout <sec>             Set the socket timeout (default 15 seconds).
  --exists-action <action>    Default action when a path already exists: (s)witch, (i)gnore, (w)ipe, (b)ackup,
                              (a)bort.
  --trusted-host <hostname>   Mark this host or host:port pair as trusted, even though it does not have valid or any
                              HTTPS.
  --cert <path>               Path to PEM-encoded CA certificate bundle. If provided, overrides the default. See 'SSL
                              Certificate Verification' in pip documentation for more information.
  --client-cert <path>        Path to SSL client certificate, a single file containing the private key and the
                              certificate in PEM format.
  --cache-dir <dir>           Store the cache data in <dir>.
  --no-cache-dir              Disable the cache.
  --disable-pip-version-check
                              Don't periodically check PyPI to determine whether a new version of pip is available for
                              download. Implied with --no-index.
  --no-color                  Suppress colored output.
  --no-python-version-warning
                              Silence deprecation warnings for upcoming unsupported Pythons.
  --use-feature <feature>     Enable new functionality, that may be backward incompatible.
  --use-deprecated <feature>  Enable deprecated functionality, that will be removed in the future.
```

pip 提供了很多命令和可用的选项。在所有这些选项当中，我们最常用的是 `pip install`。在后面加上你想要安装的库的名字，pip 就会自动帮你安装这个库。有的库还依赖其他的一些库，pip 会自动寻找合适的版本，并且安装它们。但是在那之前，你需要先配置**清华镜像源**。在你执行下面这行命令之后，pip 会默认到清华 PyPI 镜像下载包，而不是缓慢的官方源。

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

在这之后，你就可以自由地使用 pip 了。你可以安装 numpy 库，这是一个高性能计算库，我们会在下一节用到它。

```bash
pip install numpy
```

你可以尝试

```python
import numpy
```

如果没有报错，那么恭喜你，已经成功安装了自己的第一个包！

### 虚拟环境

尝试在你的机器上执行下面这段 python 代码：

```python
import sys
print(sys.path)
```

你得到了一个路径列表。许多时候我们认为 python 的包管理机制是神秘的，但是它的原理实际上相当简单。当你运行 `import xxx` 的时候，python 解释器会在 `sys.path` 包含的这些路径里寻找你要的那个包，如果没有，那就报错。路径列表中往往会包含类似 `...\\python\\lib\\site-packages` 这样的路径，这正是 pip 默认的库安装位置。打开这个路径，你会找到一个名为 `numpy` 的文件夹，刚刚运行的 `pip install numpy`，正是把 numpy 放到了这里。

默认情况下，Python 采用的是**全局包管理**模式。无论你在电脑的哪个角落打开终端运行 `pip install`，Python 的 `sys.path` 都会是一样的。当你只有一个项目时，这看起来很方便。但你很快就会遇到这种尴尬的场景：如果你正在做两个项目，项目 A 需要 `numpy==0.1`，而项目 B 需要 `numpy==0.3`，由于 `site-packages` 文件夹在同一个 Python 安装目录下只有一份，后安装的版本会直接覆盖掉前者，而更新版本的包很可能不支持旧版本的一些功能，导致项目 A 没法再跑起来。

一个更加合理的方式是，把每个项目自己依赖的包分别管理起来。我们希望能够创建一些相互隔离的场所，这样不同的项目之间就不会相互影响，这样的技术被叫做**虚拟环境**。

```shell
python -m venv .venv
```

这条命令的意思是：调用 `venv` 模块，在当前目录下创建一个名为 `.venv` 的文件夹。创建了文件夹还不算完，我们必须「激活」这个环境。

```shell
.venv\Scripts\activate
```

此时，命令提示符前面出现了一个 `(.venv)`，这说明你已经在 `.venv` 这个虚拟环境下了。再次尝试

```python
>>> import numpy
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'numpy'
```

你可以看到，之前安装的 numpy 已经没有了，这是一个隔离于外界的，不受干扰的环境。

有了前面的铺垫，你其实应该能猜出这是怎么实现的——只需要修改 `sys.path` 就好了！再次查看 `sys.path`：

```python
import sys
print(sys.path)
```

你会发现，`sys.path` 里面的 `site-packages` 已经不再是那个全局的 `site-packages`，而是被指向了 `.venv` 这个文件夹里的 `site-packages`。此时运行

```powershell
where.exe pip
```

你会发现，系统默认搜寻到的 pip 也变成了 `.venv\Scripts` 下面的这个 venv 自己独特的 pip，它会把包安装到虚拟环境的 `site-packages` 里面，而不是全局的 `site-packages`。

### Conda

在 AI 领域，很多包（如 `PyTorch`, `TensorFlow`）不仅包含 Python 代码，还深度依赖 C++ 库、CUDA 驱动等。`venv` 管不到这些非 Python 的二进制文件。我们需要比 `venv` 更加强大的虚拟环境管理。

Conda 管理的是**整个运行环境**，包括 Python 解释器本身、CUDA 工具链、C++ 编译器等。Conda 的环境通常是**全局集中管理**的。它会在你的电脑某个角落（如 `~/miniconda3/envs/`）开辟一个完整的隔离区。如果说 `venv` 靠的是修改 `sys.path` 这个 python 的包搜索路径，那么 Conda 更进一步，它还会修改 `PATH` 这样系统级的搜索路径。

Conda 是一个包管理的引擎，本体其实只有几十 MB 而已，有人围绕着它加入了各种预装的包，然后全部放在一起安装，形成了各种发行版。其中最有名，应用得也最广泛的是 Anaconda，它预装了 `NumPy`、`Pandas`、`Matplotlib`、`Scikit-learn` 等几乎所有数据科学必备的库，还提供了 Anaconda Navigator 这种图形界面，让你不用敲命令就能管理环境。在 Windows 上，很多 Python 库（比如带有 C++ 扩展的库）直接用 `pip` 装经常报错，Anaconda 预编译好了二进制文件，保证能在 Windows 上跑通。理所当然地，Anaconda 安装完的体积膨胀到了 3~5GB。

相比之下，Miniconda 提供了一个最小化的 conda 发行版，体积只有大概 300-500M，它足以满足几乎所有的日常使用。你可以到这里下载 Miniconda：[https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)。

安装完成之后，直接打开终端并输入 `conda`：

```powershell
conda
```

如果你是在 windows 上使用 conda，那么很遗憾，这个命令因为找不到 `conda` 而报错了。conda 的机制主要是通过管理环境变量来实现的，而 Windows 的环境变量管理和 Linux 不太一样，出于隔离环境变量的要求（你也不想 conda 环境里的环境变量搞坏了外面的东西），想要运行 conda，你要使用一个单独的命令提示符：

![](static/BQ1Bbg8Ago72k9x9x2sc5og7nqf.png)

如果你查看这个文件的内容，会发现它是一个非常简单的快捷方式：

```powershell
%windir%\System32\cmd.exe "/K" D:\miniconda3\Scripts\activate.bat D:\miniconda3
```

先打开一个普通的 Windows 命令行，然后在里面运行 `activate.bat` 这个脚本。

总而言之，打开这个特殊的命令行，你会发现提示符变了样子，前面多了一个 `(base)`。这指示了你当前所处的 conda 环境的名字。对于 conda 而言，默认会创建一个环境，它的名字就叫做 `base`。

```powershell
(base) C:\Users\XXX>
```

base 相对于其他环境的特殊点在于，conda 本体就安装在 base 环境。所以为了不搞出什么依赖冲突导致 `conda` 命令无法运行，通常情况下我们都会在自己建立的新环境里进行操作。

了解了这些之后，你就可以开始尝试使用 conda 了。以下是 conda 常用命令的列表：

**查询环境列表**：`conda env list` 或 `conda info --envs`。 它会列出你电脑上所有的工作区路径，当前激活的环境前面会标有一个星号。

**创建新环境**：`conda create -n <环境名> python=3.10`。建议养成在创建时显式指定 `python` 版本的习惯。如果不指定，Conda 可能会默认给一个不符合你项目要求的版本。

**激活与退出**：

- 进入环境：`conda activate <环境名>`
- 回到 base 环境：`conda deactivate`

**克隆环境**：`conda create -n <新名字> --clone <旧名字>`。

**删除环境**：`conda remove -n <环境名> --all`。

在激活了特定环境后，你就可以开始安装工具了。

**安装包**：`conda install <包名>`。 如果需要特定版本，可以使用 `conda install <包名>=1.2.3`。Conda 会自动帮你分析依赖冲突。

**更新与卸载**：

- 更新：`conda update <包名>`
- 卸载：`conda remove <包名>`

**导出与复现**：当你完成了一个项目，需要把它交给同学时，使用： `conda env export > environment.yml` 对方只需要运行 `conda env create -f environment.yml`，就能还原一个一模一样的环境。

### uv

[uv](https://docs.astral.sh/uv/) 是一个使用 rust 编写的 python 包管理器。它的一大特点是**快**。在上一节里我们介绍了 conda,如果你尝试过使用 conda 安装包，你应该体会过 conda 在解析包依赖的这个阶段运行得十分缓慢。得益于 Rust 的底层性能，uv 的依赖解析和安装速度通常比传统工具快数十倍甚至上百倍。

同时，uv 借鉴了 Rust 的 Cargo 和前端工具的设计，原生支持基于 pyproject.toml 的工作流。以往在 conda 当中，我们往往需要导出 `environment.yml`,而在 uv 的包管理模式中，所有的依赖和它们的版本要求都被清晰地写入 pyproject.toml。在 pyproject.toml 中，我们通常只声明项目的顶层依赖和较为宽泛的版本要求。而 uv 在解析这些依赖后，会生成一个极其严谨的 uv.lock 锁文件。这个文件记录了整个依赖树中每一个包的精确版本号甚至哈希值。只要基于同一个锁文件构建环境，安装的依赖分支将完全一致，这就从根本上保证了项目依赖的可复现性。

在传统的 pip 或 conda 工作流中，如果你在十个不同的项目里都用到了同一个版本的重型依赖，比如 PyTorch 或是 Transformers，你的硬盘上就会真的存下十份一模一样的庞大文件。而 uv 彻底改变了这一点。当你第一次下载某个包时，uv 会把它存放在系统的全局缓存目录中。之后无论你在多少个新的虚拟环境或是项目中需要安装这个特定版本的包，uv 会直接利用操作系统的硬链接机制，将虚拟环境中的包指向全局缓存里的同一份文件，如果你熟悉一些前端工具，你会发现这和 pnpm 是非常类似的。

由于 uv 是一个用 Rust 编译出的独立二进制文件，它的安装和运行完全不需要依赖系统中现有的 Python 环境。

在 Linux 或 macOS 系统中，你可以直接通过终端一键安装：

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果你使用 windows,也可以：

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后，我们可以体验一下现代化的项目构建流程。假设你需要新建一个研究项目，只需在目标目录下执行：

```
uv init my-research-project
cd my-research-project
```

这个命令会为你生成标准的 `pyproject.toml` 以及一个基础的项目框架。与传统方式不同，你现在不需要手动去创建或激活虚拟环境。当你需要引入科学计算或机器学习相关的重型依赖时，直接添加即可：

```
uv add torch transformers
```

执行这条命令后，uv 会在极短的时间内在后台自动为你下载合适的 Python 解释器（如果缺失的话）、创建 `.venv` 虚拟环境、解析依赖树，并通过全局缓存的硬链接完成安装。同时，它会将这两个包写入 `pyproject.toml`，并生成包含精确哈希值的 `uv.lock` 锁文件。

当你要执行代码时，直接使用：

```
uv run hello.py
```

uv 会自动接管上下文，使用当前项目的虚拟环境去运行你的脚本。如果后续你的合作者克隆了你的代码仓库，他们只需在目录中运行 `uv sync`，uv 就会严格按照 `uv.lock` 瞬间还原出与你分毫不差的底层环境。

# 第三章 NumPy, 使用计算机进行线性代数计算

在第一章中, 我们介绍了线性代数表示的神经网络计算. 而在第二章我们又学习了 Python. 大家都说 AI 是用 Python 写的. 那很好, 我们现在开始用 Python 写神经网络吧!

根据一点点程序设计的基本思想, 我们肯定希望类似矩阵乘法, 矩阵加法的操作封装好, 封装成函数, 最好还可以把向量, 矩阵用类封装(面向对象). 不过, 肯定有库实现了这些东西. 但在这里, 为了课程的继续, 我们还是尝试自己实现一下. 这里直接用列表表示向量, 嵌套列表表示矩阵:

```python
from typing import List
def matmul(mat1: List[List[float]], mat2: List[List[float]]) -> List[List[float]]:
    r1, c1 = len(mat1), len(mat1[0])
    r2, c2 = len(mat2), len(mat2[0])
    assert c1 == r2    _# 形状检查_
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

但是, 我们可以看出来, 这个实现非常麻烦. 我们需要自己检查形状等等. 最关键的是, 这个实现效率是极低的! 以一个输入层大小 784, 隐藏层大小 2048, 输出层大小 10 的 MLP 举例, 最大的矩阵乘法要处理一个(batch_size, 784)和(784, 2048)大小的矩阵相乘. 这在现代处理器上其实也没有什么压力. 我们跑一个测试(假设 batch_size 是 32):

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

这在我的机器上花费将近 3 秒(2.8159382343292236)! 这太慢了. 这还只是一个玩具级的网络(即使对我的 CPU 来说, 也是玩具级的). 造成这种缓慢的问题有很多, 感兴趣的同学可以查询相关资料. 这里我们解释为 Python 作为解释器速度天然有劣势, 并且需要维护很多对象, 列表本身是个动态还不连续的结构, 并且 Python 存在 GIL(全局解释器锁)使得程序只能单线程.

于是, 有人(一群人)专门为 Python 实现了一个高性能的线性代数库 NumPy. NumPy 是由 C 编写的, 进行了很强的优化, 然后暴露接口给 Python 调用.

虽然还没介绍 NumPy 的语法, 但是这里对比一下时间, 给大家看一下性能差距:

```python
import numpy as np
batch_size: int = 32
_# 随机生成两个矩阵_
mat1 = np.random.rand(batch_size, 784)
mat2 = np.random.rand(784, 2048)

import time
t1 = time.time()
for _ in range(1000):    # 这里因为numpy太快了所以测1000次求平均
    np.matmul(mat1, mat2)
t = time.time() - t1
print(t/1000)
```

在我的机器上, 这个时间是 0.0015103209018707275 秒, 快了 1866 倍!

> 实际情况下快不了这么多, 这里 1000 次测试缓存命中率太高了.

## NumPy 介绍

[https://numpy.org/](https://numpy.org/). NumPy 是一个很活跃的 Python 高性能计算库.

## NumPy 语法入门

NumPy 的语法尽量贴合 Python 原版的列表. 在这里, 我们先介绍基本的 NumPy 对象: NdArray.

### NdArray

ndarray 是 numpy 中用于表示向量, 矩阵, 张量的类型. 其存储在一段连续的内存上. 定义 ndarray 有许多种方法. 这里先使用最简单的.

```python
import numpy as np
vec = np.array([1, 2, 3, 4])
mat = np.array([[1, 2, 3], 
                [4, 5, 6]])
print(vec)
print(mat)
```

定义时, 可以传入 `dtype` 参数, 用于指定存储类型. 与列表不同, ndarray 中必须全部元素**是相同类型**. 如果不传入该参数, 类型将自动判断.

以下是可用的参数(这里只列出 numpy 的标准类型, 我们也推荐使用这种方式定义类型):

<table>
<tr>
<td>**对象**<br/></td><td>**含义**<br/></td></tr>
<tr>
<td>np.int8<br/></td><td>8位有符号整数<br/></td></tr>
<tr>
<td>np.int16<br/></td><td>16位有符号整数<br/></td></tr>
<tr>
<td>np.int32<br/></td><td>32位有符号整数<br/></td></tr>
<tr>
<td>np.int64<br/></td><td>64位有符号整数<br/></td></tr>
<tr>
<td>np.uint8<br/></td><td>8位无符号整数<br/></td></tr>
<tr>
<td>np.float32<br/></td><td>单精度浮点数<br/></td></tr>
<tr>
<td>np.float64<br/></td><td>双精度浮点数<br/></td></tr>
<tr>
<td>np.bool_<br/></td><td>布尔类型<br/></td></tr>
<tr>
<td>np.complex64<br/></td><td>复数<br/></td></tr>
<tr>
<td>np.str_<br/></td><td>长度为10的unicode字符串<br/></td></tr>
<tr>
<td>np.bytes_<br/></td><td>长度为10的字节字符串<br/></td></tr>
</table>

下面是一个类型定义示例:

```python
import numpy as np
vec = np.array([1, 2, 3, 4], dtype=np.int8)
mat = np.array([[1, 2, 3], 
                [4, 5, 6]], dtype=np.float32)
print(vec)
print(mat)
```

在所有类型中, 我们最经常使用的是 `float32`.

#### 定义特定 NdArray

有时候, 我们需要定义一些特殊的 array, 例如全是 0, 全是 1, 或者是我们根本不关心数据, 我们只想要一块特定形状的 array, 用来存放中间结果之类的.

定义全 0 的 array, 我们使用 `np.zeros` 函数:

```python
import numpy as np
arr = np.zeros((2, 3, 4), dtype=np.float32)
print(arr)
```

定义全 1 的 array, 我们使用 `np.ones` 函数:

```python
import numpy as np
arr = np.ones((4, 3, 2), dtype=np.uint8)
print(arr)
```

我们不关心具体的值, 只希望要一个特定的形状, 可以使用 `np.empty` 函数. 这可以减少赋值的开销.

```python
import numpy as np
arr = np.empty((3, 4, 5), dtype=np.float32)
print(arr)
```

#### NdArray 索引访问

ndarray 的索引访问几乎与嵌套列表是相同的. 但是也有一些区别.

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

`:` 的含义是"全取". 以此来实现更加灵活的访问.

#### NdArray 的计算

首先是基本的线性代数计算. NumPy 肯定是支持的.

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

numpy 还支持求最大/小值, 求最大/小值索引:

```python
import numpy as np
tsr = np.random.rand(2, 3, 4)
tsr[0][1][2] = 100
print(np.max(tsr))
print(np.argmax(tsr))
```

numpy 中可以方便转置, 转置操作是很便宜的. 因为它只会修改元信息.

```python

```

### Broadcast 广播机制

Broadcast 广播机制是一个非常重要的机制. 其基本机制十分简单.

我们知道, ndarray 的加减乘除都是要求形状匹配. 例如对应位置加减乘除需要两个数组形状一样, 矩阵乘法要求最后两个维度满足矩阵乘法的形式, 前面的维度要一致. 但是假如我们想要实现这样一个操作: 向量$\bold{v}\in \R^n$是个行向量, 矩阵$\bold{M}\in \R^{m\times n}$, 我们想让$\bold{M}$的每一行加上$\bold{v}$. 我们可能会使用 repeat 方式, 将$\bold{v}$复制成一个矩阵, 它每一行都是$\bold{v}$, 然后再相加.

但是这样其实不太好. 因为它占用了更多的存储空间, 拷贝也会有开销. 实际上我们可以让计算机直接做这件事情. NumPy 为了让用户可以实现这样的操作, 提供了广播机制.

当一个操作的输入数组形状不匹配时, numpy 会进行广播机制判定, 它的规则如下:

- 将所有输入数组的维数补全到维数最大的那个数组. 补全的方式为在前面加 1. 例如, 三个输入数组的形状分别为(2, 3), (2, 3, 4), (2, 3, 4, 5), 则会被补全成(1, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5).
- 检查每个输入数组的维度. 如果维度大小是相同的, 则没事. 如果维度大小不同, 并且其中一个数组的维度是 1, 则触发广播机制, 也是合法的. 对应维度大小为 1 的那个数组将在该维度上被广播. 否则输入不合法. 例如形状(1, 3, 4)和(5, 3, 1)是合法的一组输入. 输入形状(1, 2, 3)与(3, 2, 2)不合法.

广播机制实际上不会对数据进行复制, 因此效率比拷贝方法高, 而且内存开销也小.

### 多查资料多问 AI

NumPy 中的细节真的非常多. 而且目前的教育中, 有关矩阵的讨论是很少的. 使得多维数组的操作实际上是一门学问, 并且大家比较欠缺这方面的经验. 课程中不可能全部涉及到. 甚至说, 这里提到的相关知识, 只是 NumPy 最基础的一些概念, 而你往往需要进行一些不太寻常的多维数组操作, 你就需要去寻找 numpy 是否提供了一些接口, 你如何使用这些接口来达成你的目的. 并且, 这个部分的很多东西思考起来是很反人类的.

这就需要大家多去查资料, 多去问 AI. 很多细节并不复杂, 是一个知不知道的问题.

## NumPy 实现线性回归波士顿房价

### 线性回归

我们在高中都学习过线性回归方法. 这里介绍其更加通用的形式, 即多元的情况.

我们的场景中存在$n$个输入变量$x_i$与 1 个预测值$\hat{y}$. 对于预测值, 它满足:

$$
\hat{y} = \sum_{i=1}^n w_ix_i + b
$$

写成线性代数形式:

$$
\hat{y} = \bold{wx^T} + b
$$

其中, $\bold{w}$与$\bold{x}$都是行向量.

为了方便计算, 这里引入一些概念. 我们先将$\bold{w}$与$b$合并到一起. $\bold{w}$中添加一个元素, 其值为$b$, 而$\bold{x}$中添加一个值, 其值始终为 1. 长这个样子:

$$
\bold{w} = \begin{pmatrix}
b & w_1 & w_2 & w_3 & \cdots & w_n
\end{pmatrix};\; 
\bold{x} = \begin{pmatrix}
1 & x_1 & x_2 & x_3 & \cdots & x_n
\end{pmatrix}
$$

我们把所有数据集中样本特征拼成一个矩阵, 称之为**设计矩阵**$\bold{X}\in \R^{N\times (n+1)}$, 所有真实值组成**标签向量**$\bold{y}\in \R^N$.

预测值可以写为:

$$
\hat{\bold{y}} = \bold{wX^T}
$$

我们使用均方损失, 尝试找到最优参数$\bold{w}^*$:

$$
loss = (\bold{\hat{y}} - \bold{y})(\bold{\hat{y}} - \bold{y})^\bold{T} = (\bold{wX^T-y})(\bold{wX^T-y})^\bold{T}
$$

微分:

$$
\begin{aligned}
\bold{d}loss &= \bold{d}[(\bold{wX^T-y})(\bold{wX^T-y})^\bold{T}] \\
&= 2(\bold{wX^T-y})\bold{d}(\bold{wX^T-y})^\bold{T} \\
&= 2(\bold{wX^T-y})[\bold{d(Xw^T)}-\bold{dy^T}] \\
&= 2(\bold{wX^T-y})[\bold{d(X)w^T} + \bold{Xdw^T-dy^T}]
\end{aligned}
$$

有关$\bold{w}$的微分项:

$$
2(\bold{wX^T-y})\bold{Xdw^T} 
$$

因此, $loss$对$\bold{w}$的梯度为:

$$
\nabla_\bold{w}loss = 2(\bold{wX^T-y})\bold{X}
$$

令梯度为$\bold{0}$(类似导数为 0 求函数极值):

$$
\begin{aligned}
2(\bold{wX^T-y})\bold{X} &= \bold{0} \\
\bold{wX^TX-yX} &= \bold{0} \\
\bold{wX^TX} &= \bold{yX} \\
\bold{w} &= \bold{yX}(\bold{X^TX})^{-1}
\end{aligned}
$$

我们得到了在均方损失意义下最好的参数:

$$
\bold{w^*} = \bold{yX}(\bold{X^TX})^{-1}
$$

### 波士顿房价

波士顿房价是一个很经典的机器学习任务. 波士顿房价的任务如下:

我们都知道房价肯定和附近基础设施, 距离城市中心的距离等因素有关, 但是我们在生活中一般是定性分析的. 但是我们想要更加详细地研究房价问题. 有人统计了波士顿附近多个房子的相关属性, 每个房子收集十三个可能影响房价的特征. 特征如下:

<table>
<tr>
<td>**特征名称**<br/></td><td>**含义**<br/></td></tr>
<tr>
<td>CRIM<br/></td><td>城镇人均犯罪率<br/></td></tr>
<tr>
<td>ZN<br/></td><td>占地面积超过25,000平方英尺的住宅用地比例<br/></td></tr>
<tr>
<td>INDUS<br/></td><td>城镇非零售商业用地比例<br/></td></tr>
<tr>
<td>CHAS<br/></td><td>是否靠近查尔斯河(0代表不靠近, 1代表靠近)<br/></td></tr>
<tr>
<td>NOX<br/></td><td>一氧化氮浓度<br/></td></tr>
<tr>
<td>RM<br/></td><td>每栋住宅的平均房间数<br/></td></tr>
<tr>
<td>AGE<br/></td><td>1940年之前建成的自用住房比例<br/></td></tr>
<tr>
<td>DIS<br/></td><td>到波士顿五个就业中心的加权距离<br/></td></tr>
<tr>
<td>RAD<br/></td><td>径向高速公路可达性指数<br/></td></tr>
<tr>
<td>TAX<br/></td><td>每10,000美元的全额财产税率<br/></td></tr>
<tr>
<td>PTRATIO<br/></td><td>城镇师生比例<br/></td></tr>
<tr>
<td>B<br/></td><td>城镇黑人比例<br/></td></tr>
<tr>
<td>LSTAT<br/></td><td>低收入人口比例<br/></td></tr>
</table>

我们的目标变量是 MEDV, 含义是自住房屋的中位数价值.

### 任务分析

我们将使用线性回归方法来预测 MEDV. 首先我们定义线性回归模型的输入, 即特征向量:

$$
\bold{x} = \begin{pmatrix}
1 & CRIM & ZN & INDUS & \cdots & LSTAT
\end{pmatrix}
$$

我们定义模型的参数:

$$
\bold{w} = \begin{pmatrix}
b & w_1 & w_2 & w_3 & \cdots & w_n
\end{pmatrix}
$$

我们将数据集中所有样本的 MEDV 值组织成标签向量:

$$
\bold{y} = \begin{pmatrix}
MEDV_1 & MEDV_2 & MEDV_3 & \cdots & MEDV_n
\end{pmatrix}
$$

所有样本的特征向量组织成设计矩阵:

$$
\bold{X} = \begin{bmatrix}
\bold{x}_1 \\
\bold{x}_2 \\
\bold{x}_3 \\
\vdots \\
\bold{x}_N
\end{bmatrix}
$$

我们就可以直接计算出最优参数$\bold{w^*}$.

$$
\bold{w^*} = \bold{yX}(\bold{X^TX})^{-1}
$$

### 代码实现与分析

实现线性回归模型类.

这里我们先实现一个线性回归类:

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

我们需要加载波士顿房价数据集. 由于一些不能说的原因, 波士顿房价数据集无法在新版本的 `sklearn` 库中直接导入. 我们准备好了波士顿房价的数据集, 从本地导入.

```python
FILE = r'./boston_data.csv'
boston_data = np.loadtxt(FILE, delimiter=',', skiprows=1, dtype=np.float32)
print(boston_data.shape)
```

这个 csv 文件是这样组织的:

![](static/MNbnblOwsoYroUxWovxc433UniF.png)

我们跳过了第一行, 因为第一行是字符, 无法转化为 `np.float32`.

接下来, 我们分离特征与标签, 训练集与测试集. 这里测试集按照 20% 划分.

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

注意, 特征向量需要在最前面补 1, 进行特征扩展.

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

使用 `matplotlib` 库进行图像绘制:

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

结果如下:

![](static/MUfBbBdCnoEIFfxXZX3c0H9dnqg.png)

可以看到效果还行.

# 第四章 深度学习框架 PyTorch

我们知道, 深度神经网络基本可以通过一些数组操作来表示. 而 NumPy 提供了高效的数组操作接口. 我们确实可以使用 numpy 编写出高性能的神经网络推理与训练的程序, 但是这依然十分复杂.

- 我们需要手动推导每个参数数组的梯度公式. 也就是说, 我们的模型只要变化了, 我们就需要重新推导, 并且重新实现. 这很麻烦.
- 我们可以利用链式法则, 只实现每层的梯度计算, 随后递归调用. 但这依然不灵活, 因为你只能使用已经定义好的层, 如果你想要实现一个新层, 除非这个新层实际上只是已有层的复合层, 否则仍然需要手动实现梯度计算.

为了使得开发更加简单和灵活, 深度学习框架被构建.

## 在本章开始之前

本章开始, 你大抵是需要一台配备有 Nvidia GPU 的机器. 没有英伟达 GPU 设备, 肯定会影响你学习本章内容, 因为其中的一些代码你将无法实践. 你可以让全部运算跑在 CPU 上, 这可能会很慢. 并且, 如果你选择跑在 CPU 上, 那么你最好拥有 ≥16GB 的 RAM.

## 深度学习框架

深度学习框架可以快速地构建深度学习模型(其实不止是深度学习模型). 你只需要实现前向传播代码, 计算机就可以自动推导它的梯度计算方法. 因此你不需要考虑如何求导你的模型.

深度学习框架还实现了大量常用的模型和组件, 例如 MSE 损失函数, 交叉熵损失函数, 卷积神经网络, Adam 优化器等.

现代深度学习框架甚至还实现了分布式相关的通信操作, 方便大规模推理或者训练.

深度学习框架对于不同硬件平台做了深层次的优化, 使得其在深度学习场景下, 往往比 NumPy 更快. 并且, 其能方便调用 Nvidia GPU 等加速硬件.

## 计算图自动微分技术

### 自动微分

深度学习框架的核心是**自动微分**. 这里解释一下自动微分是什么.

我们有一个表达式$f(x)$. 它的参数只有$x$. 我们想要求$x$的微分, 朴素的思路是使用定义法: 假设我们想要求$x_0$处的微分(导数值), 我们会用下面这个式子近似:

$$
f'(x_0) \approx \frac{f(x_0 + \delta x) - f(x_0)}{\delta x}
$$

当然, $\delta x$要足够小. 使用代码实现:

```python
from typing import Callable
def differ(f: Callable[[float], float], x: float, epsilon: float=1e-6) -> float:
    return (f(x + epsilon) - f(x)) / epsilon

def f(x: float) -> float:
    return x ** 2 + 2 * x

print(differ(f, 1.))    # \approx 4
```

通过这种方法, 我们可以让计算机估算出每个变量的微分. 让计算机可以计算函数的微分, 这称之为**自动微分**.

但是这种方法有一定的缺陷. 当参数量很大时, 逐个计算是很慢的(其实也可以批量计算, 但那样又内存昂贵). 并且对于很深的嵌套函数, 会有大量重复的计算. 因此, 现代自动微分并不使用这种技术.

除了定义法之外, 我们还可以使用导函数法. 我们可以提前计算出函数的导函数. 例如, 我们知道$f(x) = x^2 + 2x$, 我们可以快速计算出其导函数为$f'(x) = 2x + 2$. 然后我们就可以代入$x$直接进行计算.

更好的是, 我们提到过, 一些函数的导函数十分简单. 例如 sigmoid 函数与 ReLU 函数. sigmoid 函数可以复用运算结果, ReLU 函数则只需要大小比较操作. 因此, 现代自动微分技术, 采用**导函数法**.

这样, 自动微分的关键就在于, 如何让计算机自动推导出算式的导函数.

### 计算图

计算图是对算式的一种建模方法, 它便于计算机处理. 计算图中有两种节点, 分别是**数据**和**操作**. 边代表输入输出, 或者说是依赖关系. 下面是一个例子.

算式$f(x, y, z) = x^2 + y^2 + x^2y^3 + z$的计算图为

![](static/B3IVbj8QxoihI0xh68vch2yWnec.png)

这非常好理解. 对于计算机来说, 我们使用面向对象的方式, 只需要实现 Node 类, 随后继承出 Variable 类和 Operation 类即可表示计算图. 我们可以使用指针方法来表示节点之间的依赖关系.

我们可以使用运算符重载的方法来捕捉算式构建时的依赖关系, 得知计算图的拓扑结构.

### 计算图与链式法则自动微分

对于复杂的算式求导是很难的, 就算是人类, 可能都很难. 但是对于单个操作求导是很简单的. 例如乘法求导, 结果就是系数; 加法求导, 结果是 1. 而算式可以看作是操作的复合. 我们可以使用复合函数求导的链式法则, 来进行求导.

对于一个由操作组成的算式来说, 对其中一个参数的导数为:

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial op_n} \frac{\partial op_n}{\partial op_{n-1}} 
\frac{\partial op_{n-1}}{\partial op_{n-2}}
\cdots
\frac{\partial op_2}{\partial op_1}
\frac{\partial op_1}{\partial x}
$$

我们可以让计算机计算出每个$\frac{\partial op_i}{\partial op_{i-1}}$, 这一般是比较简单的. 因为我们把操作拆分得很细, 往往只是一些加减乘除. 随后我们将每一项相乘, 即可得到我们需要的结果.

至于操作之间的依赖关系, 我们可以利用计算图获得. 下面用一个例子来说明.

我们就直接拿计算$f(x, y, z) = x^2 + y^2 + x^2y^3 + z$在$(1, 2, 3)$处的梯度为例. 我们将利用链式法则进行求解.

我们观察计算图, 从结果向输入参数推导. 首先我们需要计算操作$+$的导数. 其写出来为:

$$
\begin{aligned}
f &= add(x^2 + y^2 + x^2y^3, z) \\
\frac{\partial f}{\partial z} &= 1 \\
\frac{\partial f}{\partial (x^2 + y^2 + x^2y^3)} &= 1

\end{aligned}
$$

我们可以发现, 加法操作对两个输入参数求导, 其实就是 1. 而我们在运行计算图时, 可以将加法操作的输入参数存储起来(而不是运算完之后抛弃). 这样就不需要存储一个很大的式子. 只需要存储数值.

现在, 我们知道了$\frac{\partial f}{\partial z} = x^2 + y^2 + x^2y^3 + 1$, 我们继续求$x$与$y$.

$$
\begin{aligned}
x^2 + y^2 + x^2y^3 &= add(x^2 + y^2, x^2y^3) \\
\frac{\partial(x^2 + y^2 + x^2y^3)}{\partial (x^2 + y^2)} &= 1 \\
\frac{\partial(x^2 + y^2 + x^2y^3)}{\partial x^2y^3} &= 1
\end{aligned}
$$

接下来我们研究$x^2 + y^2$.

$$
\begin{aligned}
x^2 + y^2 &= add(x^2, y^2) \\
\frac{\partial (x^2 + y^2)}{\partial x^2} &= 1 \\
\frac{\partial (x^2 + y^2)}{\partial y^2} &= 1
\end{aligned}
$$

$x^2$与$y^2$都是经过$power$操作得到的:

$$
\begin{aligned}
x^2 &= power(x, 2) \\
\frac{\partial x^2}{\partial x} &= 2x \\
y^2 &= power(y, 2) \\
\frac{\partial y^2}{\partial y} &= 2y
\end{aligned}
$$

别忘记了, 我们现在只计算了$x^2 + y^2$路径得到的导数值. 还需要计算$x^2y^3$的导数值:

$$
\begin{aligned}
x^2y^3 &= multiply(x^2, y^3) \\
\frac{\partial x^2y^3}{\partial x^2} &= y^3 \\
\frac{\partial x^2y^3}{\partial y^3} &= x^2

\end{aligned}
$$

有关$\frac{\partial x^2}{\partial x}$的导函数, 我们已经计算过了. 我们不会重复计算. 接下来计算$\frac{\partial y^3}{\partial y}$.

$$
\begin{aligned}
y^3 &= power(y, 3) \\
\frac{\partial y^3}{\partial y} &= 3y^2
\end{aligned}
$$

我们手工计算出了所有项. 现在我们使用链式法则合并它们

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

将$(1, 2, 3)$代入, 得到:

$$
\nabla f = \begin{bmatrix}
18 \\
16 \\
1
\end{bmatrix}
$$

我们是手工计算, 一直采用符号推理. 但是计算机实现时, 实际上只保存了数值. 计算机并不是先把每个微分项推导出来, 再全部乘起来, 而是一边推导一边乘.

在实际实现的过程中, 算法会对计算图进行**反向传播**, 按照一定顺序遍历计算图. 不断累乘梯度值, 当遍历完成时, 也就求出了每个参数的导数值.

## PyTorch 语法入门

PyTorch 是目前最常用的深度学习框架. torch 中的基本类型是 `torch.Tensor`, 与 NumPy 中的 `numpy.ndarray` 对标. 相关语法与 `ndarray` 几乎相同.

### Tensor 张量

PyTorch 中的张量运用方式与 NumPy 中的 ndarray 数组大量雷同(为了方便迁移). torch 中的张量使用 torch.tensor 定义:

```python
import torch
tsr = torch.tensor([1, 2, 3], dtype=torch.float16)
print(tsr)
```

tensor 可以通过 device 参数指定 tensor 所在的设备.

> 设备: 对于很多异构计算结构的机器来说, 它不止拥有 CPU, 还有 GPU, NPU, TPU 等异构加速设备. 这时候你定义张量, 就需要指定是在哪个设备上定义的. 因为, 往往来说, 不同设备之间不会共用存储器.
> 并且, 你需要指明一个操作是在哪个设备上完成的. 例如, 你要如何指定一个矩阵乘法操作在编号为 0 的 GPU 上执行? torch 的方法是, 执行操作前, 判断输入张量属于哪个设备, 你应该让所有输入张量属于同一个设备. 当所有输入张量都属于同一个设备, 那么这个操作肯定就是在这个设备上执行的.

```python
import torch
tsr = torch.tensor([1, 2, 3], dtype=torch.float16)
tsr_cuda0 = torch.tensor([1, 2, 3], dtype=torch.float16, device='cuda')
tsr_cuda1 = torch.tensor([1, 2, 3], dtype=torch.float16).to('cuda')
try:
    tsr + tsr_cuda0
except Exception as e:
    # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
    print(e)
# tensor([2., 4., 6.], device='cuda:0', dtype=torch.float16)
print(tsr.to('cuda') + tsr_cuda1)
```

_注意, 运行示例代码需要配备 CUDA 加速硬件的设备._

### nn.Module 模块

`torch.nn.Module` 是 PyTorch 的核心组件. 它是对模型中模块的抽象建模(模型本身也可以看作一个 Module). 类似于函数, `nn.Module` 有输入与输出, 并且拥有参数. 它实现了 `__call__` 魔术方法, 因此可以像函数一样使用.

继承 `nn.Module` 后, 你不需要实现 `__call__` 方法, 你需要实现 `forward` 方法. 例如:

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()    # 先执行父类的init
        self.n = n
        
        _# 你可以使用nn.Parameter来包装模型的参数_
        self.weights = nn.Parameter(torch.randn(n))
        
        _# 使用register_buffer方法来注册不需要更新的张量_
        self.register_buffer('no_grad_tensor', torch.randn(n))
        
        """
        推荐使用nn.Parameter和register_buffer来定义数据. 
        防止在Module嵌套关系复杂之后, torch参数推导出现问题.
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights + self.no_grad_tensor

net = Net(128)
x = torch.randn(128)
y = net(x)
print(y, y.shape)
```

你可以使用 `parameters` 方法来获取 `nn.Module` 中所有注册的参数(返回的是生成器形式). 如果你还需要模型的参数注册时的标识符, 你可以使用 `named_parameters` 方法:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n))
        self.w2 = nn.Parameter(torch.randn(n))
        self.b = nn.Parameter(torch.randn(n))
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * self.w1 + x2 * self.w2 + self.b

model = Model(128)
print(type(model.parameters()))
for p in model.parameters():
    print(p)

print(type(model.named_parameters()))
for p in model.named_parameters():
    print(p)
    print(p[0]) _# 使用p[0]来访问参数名_
```

模型可以整体转移到某个 device 上. 使用 `to` 方法:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n))
        self.w2 = nn.Parameter(torch.randn(n))
        self.b = nn.Parameter(torch.randn(n))
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * self.w1 + x2 * self.w2 + self.b

x = torch.randn(128, device='cuda')
y = torch.randn(128, device='cuda')
model = Model(128)
try:
    model(x, y)
except Exception as e:
    print(e)

model.to('cuda')    # 转移到cuda加速设备上

z = model(x, y)
print(z.device, z.shape)     # cuda:0 torch.Size([128])
```

### 损失函数

PyTorch 中定义损失函数比较灵活. 比较推荐的方式是使用 nn 中自带的一些损失函数(其本质是 Module):

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()    # 比较常用的交叉熵损失
logits = torch.randn(4, 128)    # shape: (batch_size, features)
targets = torch.tensor([0, 99, 67, 2])    # batch中每个正确类别的索引

loss = criterion(logits, targets)
print(loss.item())    # 如果不使用item方法, 则返回的是tensor(xxx), 即tensor包装的一个数, 0维

criterian = nn.MSELoss()    # MSE是均方损失
logits = torch.randn(4, 128)
targets = torch.randn(4, 128)    # 均方损失函数的targets输入不是索引
loss = criterian(logits, targets)
print(loss.item())
```

其实, 你不用 nn 自带的也是可以的. 只要是运算就行. 您乐意用一个函数包装还是不乐意, 乐意用 `nn.Module` 包装还是不乐意, 乐意啥都不干直接写出来还是不乐意, 都看您:

```python
import torch
import torch.nn as nn

class CustomLoss(nn.Module):    # 使用nn.Module, 最正统
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean((predictions - targets) ** 2)

def custom_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 用函数包装, 还凑合
    return torch.mean((predictions - targets) ** 2)

predictions = torch.randn(32, 128)
targets = torch.randn(32, 128)

criterion = CustomLoss()
loss = criterion(predictions, targets)
print(loss.item())

loss = custom_loss(predictions, targets)
print(loss.item())

loss = torch.mean((predictions - targets) ** 2)    # 直接写出来, 不卫生
print(loss.item())
```

### 自动求导

torch 中根据某个值反向计算出梯度, 使用 `backward` 方法. 我们一般对 loss 使用 `backward` 方法. 但是这并不意味着其它张量不可以. loss 与其它张量也并没有什么实现上的不同.

```python
import torch
W = torch.randn(3, 4, requires_grad=True)
b = torch.randn(4, requires_grad=True)
x = torch.randn(2, 3)
y = torch.randn(2, 4)

pred = x @ W + b

loss = torch.mean((pred - y) ** 2)
loss.backward()
print(W.grad)
print(b.grad)
```

> 有关张量 `requires_grad` 参数: 该参数用于指定张量在反向传播时是否需要计算梯度.
> 修改该参数有很多种方式, 最暴力的方式是直接修改:
>
> ```python
> ```

import torch
x = torch.randn(4)
x.requires_grad = True

```
> 一般来说, 张量在创建时, 可以指定是否需要梯度, 在创建张量的函数种一般会带有一个参数requires_grad. 你可以在定义张量时指定. 
> ```python
import torch
w_no_grad = torch.randn(2, 3)
w_with_grad = torch.randn(2, 3, requires_grad=True)

print(w_no_grad.requires_grad, w_with_grad.requires_grad)
```

> 被 nn.Parameter 包装的张量, 一般来说它的 requires_grad 是 True(会自动设置张量的 requires_grad 为 True).
>
> ```python
> ```

import torch
import torch.nn as nn
w_no_grad = torch.randn(2, 3)
w_param = nn.Parameter(torch.randn(2, 3))

print(w_no_grad.requires_grad, w_param.requires_grad)    # False True

```
> 你可以使用`requires_grad_`方法来修改单个张量, 或者一个Module的参数是否需要梯度:
> ```python
import torch
x = torch.randn(4)
x.requires_grad_(True)

model = Model()
model.requires_grad_(False)
```

### Optimizer 优化器

优化器用于优化一组参数. 创建它需要输入被优化的参数. 这边使用比较朴素的 SGD 优化器举例(SGD, 随机梯度下降). 首先, 你需要计算每个参数的梯度值(一般使用 `backward` 方法), 随后优化器使用 `step` 方法进行一步优化.

下面是一个示例:

```python
import torch
from torch.optim import SGD
iteration = 100
epoch = 10
optimizer = SGD(model.parameters(), lr=1e-3, momentum)
for _ in range(epoch):
    for i in range(iteration):
        x, label = data[i]
        pred = model(x)
        loss = loss_function(pred, label)
        
        loss.backward()
        optimizer.step()
```

## Quick Start

# 第九章 大语言模型 LLM

## 分词器 Tokenizer

Tokenizer 是 LLM 的基础部件之一，其目的是将 raw text(通常以 Unicode strings 表示)转换为模型可以处理的数据。由于通常情况下模型只能处理数字，因此 Tokenizer 所发挥的作用就是将我们的 raw text 转为 Interger ID(**encode**)，其中每个 Interger 代表一个 Token，并将这些 tokens 解码成 Unicode strings(**decode**)

Tokenizer 的 **vocabulary size** 就是所有可能出现的 token 的个数和

### Tokenizer 的类型

#### Word-based Tokenizer

当我们想对一个句子做切分，第一个出现在我们脑子里的想法就是将原始文本切分(spilt)为单词，例如

特点：保留了完整的语义，但是会导致词表极其庞大，且容易遇到 OOV(Out-of-Vocabulary)问题

#### Character-based Tokenizer

既然 Word-based Tokenizer 会出现词表过大和 OOV 问题，那么我们用一些 mate 的元素填充词表不就可以规避这个问题了吗————这个自然的想法带来了 Character-based Tokenizer。如果我们按照单个字母或字符切分，就能得到一个词表很小（只有 26 个字母）且不会遇到 OOV 问题的 Tokenizer（这里不考虑特殊字符和表情等）

那么代价是什么呢？

特点：单个字符缺乏语义，使序列变的非常长，增加计算成本

#### Subword-based Tokenizer

自然而然的，人们想到了在上面两种方式之间 trade-off 以集成上面两种 Tokenzier 的优点，Subword-based Tokenizer 诞生了，这是目前的 LLM 的主流做法。它既让常见的单词在词表中占有一席之地，也可以做到将没见过的词回退到 word pieces 或 character

特点：将词汇拆分成了较小的、有意义的片段

下面，我们将进入 Tokenizer 的训练环节，告诉大家如何得到一个自己的 Tokenizer

### Tokenizer 的训练
