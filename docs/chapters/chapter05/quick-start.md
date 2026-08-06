---
title: "Quick Start"
order: 5
---

# Quick Start

下面，我们使用一个例子来快速了解一下一个PyTorch深度学习应用的组成。熟悉PyTorch。

之前我们搞过线性回归波士顿房价。我们在第三章的实践中（如果你做了实践），我们使用numpy实现了一个手写数字识别任务的训练。现在我们使用PyTorch实现一个相同的任务。

## 定义模型

首先我们要定义模型（本质是一个nn.Module）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 手写数字图片大小是 28x28，展平后是 784 维的一维向量
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
```

## 数据准备与组织

现在我们需要准备数据。我们直接使用`torchvision`来下载MNIST数据集。

```python
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

我们获取到了一个torch中的`Dataset`类对象`train_dataset`。接下来我们使用`dataloader`来包装它。

```python
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
```

## 实例化模型，损失函数与优化器

接下来我们定义模型，损失函数与优化器。做好训练准备。

```python
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## 加速设备识别与迁移

我们经常需要识别所运行的平台拥有什么样的加速硬件，然后再将模型等张量迁移到设备上来调用硬件加速计算能力。如果未检查到硬件，我们应该回退到cpu上（提升代码鲁棒性）。

```python
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"已挂载 GPU: {torch.cuda.get_device_name(0)}")

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("已挂载 Apple MPS 加速")

    else:
        device = torch.device('cpu')
        print("未检测到加速硬件，使用 CPU 进行计算")

    return device

device = get_device()
model.to(device)
```

下面是训练代码：

```python
from tqdm import tqdm
epoch = 5

for _ in range(epoch):
    model.train()
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test ]')
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\n>>> Epoch {epoch+1} Summary: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
```

## 模型保存

一般来说，我们保存模型时保存该模型的`state_dict`。`state_dict`是一个Python字典，保存了模型的“状态”。这个状态包含了模型的结构，以及这个结构的参数。例如，我们如果查看刚刚训练的模型的`state_dict`，是这样的：

```python
print(model.state_dict())
```

不只是模型，优化器也是可以保存`state_dict`的。

```python
print(optimizer.state_dict())
```

保存`state_dict`可以使用`torch.save`函数，其本身会调用`pickle`进行序列化保存。


```python
PATH = r'./model.pt'
torch.save(model.state_dict(), PATH)
```




## 加载模型进行推理

模型的参数与模型的“类”是分开保存的。所以需要向一个创建好的对象导入`state_dict`。不过在此之前，先要用`torch.load`把路径中的文件读出来，变成Python字典对象：

```python
model = MLP()
state_dict = torch.load(PATH)
mode.load_state_dict(state_dict)
```
