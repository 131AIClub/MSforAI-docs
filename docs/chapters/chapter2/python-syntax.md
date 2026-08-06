---
title: "Python 语法入门"
order: 4
---

# Python 语法入门

## Hello World

在终端直接输入 `python` 进入终端交互模式：

```bash
PS C:\Users\MSforAI> python
Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

\>\>\> 是输入提示符。在这后面输入 print("Hello, world!")：

```bash
Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> print("Hello, world!")
Hello, world!
>>>
```

> 退出交互模式：输入 `exit()`。即调用退出函数。

不过 Python 肯定不止有这种在终端里输入一行，解释一行的模式。Python 更常用的是使用脚本。

创建一个后缀为 `.py` 的文件（Python 源文件的后缀是 `.py`），内容为：

```python
print("Hello, world!")
```

> Python 不需要什么 main 函数作为程序入口。程序默认从第一行开始执行。Python 语句之间使用换行符\n 作为分隔，而不使用分号；进行分隔。

在当前目录下的终端中输入 `python xxx.py`（xxx 是你的文件名字），就可以运行。

```bash
PS C:\Users\MSforAI\Desktop> python main.py
Hello, world!
```

## 变量与运算

### 变量赋值

Python 中变量采用等于号 `=` 进行赋值，并且**不使用**类型声明：

```python
a = 123
var = 0.114514
b = "Hello, MS for AI!"
```

这是因为 Python 是一门“动态类型”语言。

**动态类型语言**：程序在运行时才去检查类型。例如：
```python
a = 1
b = "2"
c = a + b
```
这个Python程序会报错。因为`a`与`b`一个是整数类型，一个是字符串类型，不能执行`+`操作。Python在运行程序后，会在内部隐式自动标注变量的类型，只有运算的时候出现问题了，才会报错。

Python中能实现随便修改类型的操作：

```python
var = "114514"
var = 1919810
```

这个操作是允许的。程序结束前，`var` 的值最后是 1919810。

Python 中支持多个变量同时赋值，例如：

```python
a = b = 123
a, b, c = 1, 1.2, "3"
```

### 变量基本类型

Python 中的变量有这些标准类型：

- 数字 Numbers
- 字符串 String
- 列表 List
- 元组 Tuple
- 字典 Dictionary
- 布尔值 Bool

#### 数字

数字用于存储数。它们也有以下四种类型：

- `int` 有符号整数
- `long` 长整型，也可以代表八进制和十六进制**（python3 后不存在 long，与 int 类型合并）**
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

> #是 Python 的注释标识符。从#开始直到换行会被视为注释内容，不会被 Python 解释器运行。

#### 字符串

字符串可以包含大多数字符：

```python
s = "1a_-%$@MSfor AI! 你好, 世界!\n\t"
```

你也可以定义空字符串：

```python
s = ""
```

字符串也可以用单引号定义。与双引号没区别。这样可以方便地在字符串中内含"：

```python
s = 'abc'
s = '她说: "你是好人, 但是...你没有学过MS for AI. 抱歉!"'
```

定义很大的字符串，可以使用 3 个双引号：

```python
s = """Missing Semester for AI
copyright@131AIClub
人工智能教育中缺失的一课
这是一个很大大大大大大大的字符串...
没有了
"""
```

注意，"""也可以用于大规模注释：
```python
"""
这是一个很大大大的注释
一般会用于编写文档.
"""

"""这也是注释, 换不换行并不会影响"""
```

`len` 函数可以用于获取串的长度：
```python
s = "012345"
length = len(s)    # 6
```

字符串的一个很重要的操作是**切片**。切片使用中括号来标记一个下标区间，左闭右开：

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


字符串切片还可以有第三个值，含义是“步长”，即取数每一步的长度。默认为 1，即区间内全取。步长也可以取负数。

```python
s = 'Missing Semester for AI'
print(s[0: 7: 2])    # Msig
print(s[8: 16: 3])    # See
```

切片的三个值都是可以不输入的。默认值分别为：

- 第一个值（起始位置）：0，即串的开头。
- 第二个值（结束位置）：len(s)，即串的长度。
- 第三个值（步长）：1，即全取。

`

例如：

```python
s = 'Missing Semester for AI'
print(s[::])    # Missing Semester for AI
print(s[:: -1])    # IA rof retsemeS gnissiM
print(s[2:])    # ssing Semester for AI
print(s[: 3:])    # Mis
```

#### 列表

List（列表）是 Python 中使用最频繁的数据类型。

列表可以完成大多数集合类的数据结构实现。它支持数字，字符串甚至可以包含列表（即嵌套）。

准确来说，列表几乎可以装任何对象。并且不要求列表元素是同一个类型。

```python
l = [1, "2", [3, 4]]
l = []    # 空列表
l = list()    # 这个也是空列表
```

列表是动态长度的，可以通过 `append` 方法来添加新元素：

```python
l = []
l.append("I")
l.append("love")
l.append(['MSforAI', '!'])
print(l)    # ['I', 'love', ['MSforAI', '!']]
```

列表删除元素可以使用 `pop` 和 `remove` 方法。

- `pop` 方法接收一个参数作为要删除的元素的索引。缺省值为-1，即最后一个元素。`pop` 方法会返回删除的元素：

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

- `remove` 方法接收一个匹配值，会删除第一个匹配到的元素。`remove` 没有返回值（返回 `None`）：

```python
l = [1, 2, 'MSforAI', '天天天国地狱国']
remove_value = l.remove('MSforAI')
print(remove_value)    # None
print(l)    # [1, 2, '天天天国地狱国']
```

列表可以使用索引访问和修改元素：

```python
l = [1, 2, 3]
print(l[1])    # 2
l[0] = 'new'
print(l)    # ['new', 2, 3]
```

列表支持切片操作。逻辑与字符串基本相同：

```python
l = [0, 1, 2, 3, 4, 5]
s = l[::-1]    # [5, 4, 3, 2, 1, 0]
s = l[:2]    # [0, 1]
```

#### 元组

在当前阶段，你可以认为元组与列表最大的区别就是：元组是不可变的。

定义元组使用括号()：

```python
t = (1, 'it', 'is a tuple!')
t = (,)    # 空元组
t = tuple()    # 空元组
```

> 为什么 `t=(,)` 要加逗号？
> 这是因为，如果你写()会出现歧义：为什么()不是个空表达式？虽然 Python 解释器确实会认为()是个空元组，但是为了避免歧义，最好写个逗号。
> 实际上，你在定义只有一个元素的元组时，必须要写个逗号：
>
> ```python
> t = (1,)
> ```
> 如果你不写逗号，Python解释器会认为这是个表达式，最终`t`的类型是`int`
> ```python
> t = (1)
> print(type(t))    # <class 'int'>
> ```

你可以通过索引来读取元组中的值，但是你不能修改（准确来说，是你不能更换元组中的对象）：

```python
t = (1, 2, 3)
print(t[2])    # 3
t[0] = 123    # 这行会报错

t = (1, 2, [1, 2])
t[2].append(3)    # 可以!
print(t)
```

这里列表可以被更新是因为，你没有更换元组中的对象，列表还是那个列表，只是列表自己的内含值变了。

由于元组不能修改，所以元组是定长的。

元组也支持切片操作，逻辑类似字符串：

```python
t = (1, 2, 3, 4, 5)
print(t[:3])    # (1, 2, 3)
```

#### 字典

字典是无序的对象集合。由键值对组成。你可以通过键 key 来快速查找到值 value。

> 字典在实现上使用哈希表（散列表）。

字典使用花括号{}定义，使用引号`:`来分隔键值对：

```python
d = {1: '1', 2: '4', 3: 'AI', 4: 'vibe'}
d = {}    # 空字典
d = dict()    # 空字典
```

字典的值可以是绝大多数的 Python 对象。但是字典的键必须是**可哈希的**。列表与元组是不可哈希的，因此不能作为字典的键。

> Python 判断是否可哈希，是通过该对象是否实现 `__hash__` 魔术方法来判断的。这部分的内容，有兴趣的同学可以自己查询资料来学习。在后续学习了面向对象后，大家对这一块会有更清楚的认识。

```python
d = {
    (1, 2, 3): 654    # 报错!
}
```

字典可以通过键（key）来查找值（value）：

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

字典添加键值对最常用以下两种方式：

- 直接赋值
- `update` 方法

直接赋值的形式如下：

```python
d = {}
d['key1'] = 'value1'
d['key2'] = 'value2'
d[131] = ('ai', 'club')
print(d)    # {'key1': 'value1', 'key2': 'value2', 131: ('ai', 'club')}
```

`update` 方法形式如下：

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

删除字典的键，可以使用 `del` 语句，或者 `pop` 方法。

更常用的是 `pop` 方法，`pop` 方法会返回删除值：

```python
d = {'club': 131, 'ms': ['y', 'e', 's'], 'key': 100}
value = d.pop('ms')
print(value)    # ['y', 'e', 's']
print(d)    # {'club': 131, 'key': 100}
```

`del` 语句的形式：

```python
d = {'club': 131, 'ms': ['y', 'e', 's'], 'key': 100}
del d['ms']
print(d)    # {'club': 131, 'key': 100}
```

> `del` 语句还有更多的用处。这里限于课程性质，不介绍。想要学习的同学可以查找相关资料来学习。
> C++ 中 delete 语句很重要，而在 Python 中 `del` 语句却没那么重要。这是因为 Python 中有**垃圾回收机制**。** **Python 解释器维护了一个垃圾收集器（Garbage Collector，GC），它会按照一个规则（通常是引用计数）来回收用户不使用的对象。因此 Python 程序员一般情况下不需要考虑回收问题。

#### 布尔值

Python 中的布尔值类似数字型变量。其只有两种值：`True` 和 `False`：

```python
# 布尔值要大写: 是True和False, 不是true和false.
t = True
f = False
```

### 基本运算

#### 数字型变量加减乘除

一些大家肯定懂的东西，这里就快速介绍。

数字型之间的加减乘除是很符合直觉的：

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

复合运算符，即 `+=`，`-=` 等，Python 也是支持的：

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

接下来介绍一些特殊的运算：

#### 字符串加法与乘法

字符串的加法实际就是拼接：

```python
s1 = 'Hello, '
s2 = 'world!'
print(s1 + s2)    # Hello, world!

s = '131' + 'AI' + 'Club'
print(s)    # 131AIClub
```

字符串乘法就是重复拼接：

```python
s = '131' * 3
print(s)    # 131131131
```

#### 列表加法与乘法

列表加法与乘法与字符串类似：

```python
l1 = [1, 2, 3]
l2 = ['1', '2', '3']
l = l1 + l2
print(l)    # (1, 2, 3, '1', '2', '3')

l = [1, 3, 1] * 3
print(l)    # (1, 3, 1, 1, 3, 1, 1, 3, 1)
```

#### 元组的加法与乘法

与列表，字符串基本相同：

```python
t1 = (1, 2, 3)
t2 = ('1', '2', '3')
t = t1 + t2
print(t)    # [1, 2, 3, '1', '2', '3']

t = [1, 3, 1] * 3
print(t)    # [1, 3, 1, 1, 3, 1, 1, 3, 1]
```

#### 字典的合并运算 |

字典的合并运算 `|` 是在 Python3.9 引入的。它可以合并两个字典：

```python
d1 = {1: '1', 2: '2'}
d2 = {'1': 1, '2': 2}
d = d1 | d2
print(d)    # {1: '1', 2: '2', '1': 1, '2': 2}
```

## 分支与循环

程序的基本控制结构就是顺序，分支，循环。Python 作为健全的语言，肯定是可以实现这些基本的控制结构。

### 分支

#### if-else

Python 实现分支需要使用 `if` 与 `else` 关键字：

```python
if True:
    a = 1
else:
    a = 2
print(a)    # 1
```

注意程序缩进：分支的语句体需要向内缩进一个单位（一般是 4 个空格）

多分支可以使用 `else if`：

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

注意，在多分支结构中，遇到了一个匹配项，则后续全部分支都会跳过。

#### match-case

Python3.10 引入的新语法，对标 switch 语法。

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

留个印象就可以了，很少用。

### 循环

循环主要由两种关键字实现，分别是 `while` 和 `for`。

#### While

`while` 关键字后面需要写一个布尔值表达式，代表循环继续条件。循环会一直进行到条件不满足为止：

```python
i = 0
while i < 10:
    print(i)
    i += 1

# 这是一个死循环
while True:
    pass
```

如果想要中途直接退出循环（而不是等待循环条件不满足），可以使用 `break` 语句：

```python
# 这个循环只会输出到6
i = 0
while i < 10:
    print(i)
    if i > 5:
        break
    i += 1
```

还有一个语句是 `continue`。它会直接开始下一次循环，跳过 `continue` 后面执行的语句：

```python
i = 0
while i < 131:
    print(i)
    continue    # 这里会导致死循环!
    i += 1
```

`while-else` 语句。实际上 `while` 还有一个 `else` 语句，不过大多数情况下我们是不使用的。它的含义就是在 `while` 循环结束后，执行 else 语句中的内容。但是注意：通过 `break` 退出的循环，不会执行 `else` 的内容。

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

#### For

`for` 比 `while` 用的会更多一些~~（我的身边统计学）~~。`for` 循环需要指定循环变量与迭代器。`range()` 函数会生成一个迭代器。它接收整数作为输入，返回一个按照一定规则迭代整数的迭代器。列表与元组也是可迭代的，会依次按顺序迭代出其内含的元素。

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

循环变量有时候我们是不关心的，即在循环中我们不会使用。例如说，我们只是希望一个相同的过程执行 100 次，但是我们并不需要知道我们执行到第几次。而且，最关键的是，我们懒得想变量名。这个时候，你可以用占位符来替代变量标识符：

```python
for _ in range(100):
    print("I don't care i.")
```

除了替换循环变量标识符，占位符也可以替换变量赋值的位置：

```python
t = ('131AIClub', 'MSforAI', 114514)
name, coure, _ = t    # 我不关心最后一个值, 而且我懒得想变量名
```

在终端交互环境中，占位符 `_` 可以用于指代最近的一个表达式的值：

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

有关迭代器的更多讲解，介于课程性质，不介绍。感兴趣的同学可以自己查阅资料学习。

## 函数

Python 使用 `def` 关键字声明函数。分别需要声明：函数标识符（函数名），函数参数（arguments）。并使用 `return` 语句来声明返回值。如果没有声明返回值，则返回 `None`。

```python
def func(x, y):
    z = x + y
    return z

def foo():
    pass
```

通过函数标识符 +()来调用函数：

```python
def func(x, y):
    return x + y

print(func(1, 2))    # 3
print(func(x=1, 2))    # 报错! 关键字指定要在最后
print(func(1, y=2))    # 3
print(func(x=1, y=2))    # 3
```

调用函数时，不写参数名的是**位置参数**，写参数名的是**关键字参数**。

### 递归

Python 是支持递归的，例如下面这个计算 Fibonacci 数的程序：

```python
def fibonacci(n: int) -> int:
    if n <= 2:    return 1
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(1, 11):
    print(fibonacci(i), end=' ')
```

> Python 类型注释。Python 虽然不需要声明变量类型，但是你也可以通过类型注释来声明变量类型。类型注释在运行时不会有任何影响，但是在编写程序时的 lsp 分析中很有用：lsp 可以进行类型推导，提前发现一些类型错误。
> 类型注释的基本方法是在变量的后面加 `:<类型>`，例如 `x: int`，`s: str`。函数返回值的类型注释可以通过在函数声明行添加 `-><类型>` 来实现，例如 `def f() -> None:`，`def g(x: int) -> str:`

### 闭包

闭包。Python 函数的参数与返回值可以是函数！

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

> 可变参数 `*args` 与 `**kwargs`：`*args` 用于接收任意数量的位置参数，这些参数会被收集到一个**元组**中。而 `**kwargs` 用于接收任意数量的关键字参数，会被收集进一个**字典**中。

闭包的实际定义其实比较复杂，并不是简单的“参数与返回值可以是函数”。这一块内容其实不能算是 Python 入门了……

大家只要有这个思想就可以了，知道 Python 中函数是一个比较灵活的东西，实际上只是一个 `Callable`，可调用的东西。

## 面向对象

Python 是一门面向对象的语言。尽管一直有人在抨击 OOP（面向对象英文简写）模式，但是面向对象确实是利于人类程序员建模现实问题，组织程序结构。不管你有没有对象，你都得先想办法面向对象。每当我写一个新项目时，第一个写的总是继承抽象类的类，不管有没有必要写这个类。但是，最佳实践这一块 hhh。

简单回顾一下面向对象的一些基本概念。类是对象的蓝图或模板，对象是类的实例。类定义了对象的属性和行为。我们把一些事物分类进一个类中，并且抽象出一些共同属性，就构成了类。

例如说，我想要编写一个服务器后端，它的功能是接收用户请求，返回用户需要的图片（一个很常见的例子是图床，虽然那个是随机的）。我其实可以选择为用户请求实现一个类，每当接收一个用户请求，我就实例化一个用户请求对象。而用户请求类内部包含请求的用户名，请求的图片 id，请求的优先等级等属性，这些属性是共通的。如果我不适用类来实现，那这将变得很麻烦，我不得不维护很多的变量，没有办法把这些属性组织起来。

Python 中定义类使用 `class` 关键字：

```python
class A:
    ...

a = A()    # 实例化一个对象
```

不过现在这个类啥都没实现……

一般来说首先要实现的是构造函数。Python 中规定了类的构造函数为 `__init__` 魔术方法：

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

`self` 不需要外部提供，代表对象本身。这个参数是不能放在中间，只能放在第一个的。如果没有 `self` 参数，该方法被称为静态方法（static method）。静态方法需要 `staticmethod` 装饰器才能实现。

### 继承与多态

Python 类当然是支持继承与多态的。

类的继承形式如下：

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

类可以多层继承，也可以多重继承：

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

Python 的类默认继承 `object`：

```python
class A(object):
    ...

class A:    # 等价
    ...
```

如果你想在子类中扩展而非完全替换父类的方法，你可以使用 `super()` 函数。`super()` 将返回一个类似父类对象的东西：

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

多态即同一个接口由多个不同类型使用，实现不同的功能。例如：

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

这里都继承了 `area` 接口，但每个子类有自己的实现。以此来实现多态。

### 类型检查

Python 可以使用 `isinstance` 与 `issubclass` 进行类型检查：

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

`issubclass` 用于检查一个类是否属于某个类的子类：

```python
class Animal:    pass
class Dog(Animal):    pass

print(issubclass(Animal, Dog))    # False
print(issubclass(Dog, Animal))    # True
```
