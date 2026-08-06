---
title: "安装与环境配置"
order: 3
---

# 安装与环境配置

在安装 Python 之前，先要说明：

- 你们之后实际上很少用全局环境的 Python，也就是你待会要安装的 Python。本课程后面会讲解 Python 虚拟环境相关的内容。在那之后实际上大家就使用 conda 与 uv 创建的虚拟环境中的 Python 了。
- 有关环境配置中的 VsCode 相关内容。这里假定你们已经安装了 Visual Studio Code。注意是 Visual Studio Code，而不是 Visual Studio。后者就是在程序设计课上要求安装的。

## Windows

不要从什么应用商店下！！！

前往 Python 官网（python.org），找到 Downloads 页面：

![](/static/UTMUbLLNIoQJ8exXsSCcXdWonRd.png)

然后直接点击“Download Python install manager”：

![](/static/RoYEbCGQboDY7hxwwrqcHZCenpf.png)

国内网络环境下载会比较慢……

[施工，加速站点]

下载好之后，双击运行该文件。在弹出的窗口中点击确认。

[施工]

## MacOS

### 安装 Homebrew（Mac 核心包管理器）

为了绕过网络环境限制，使用了国内维护的安装脚本，并选择了更稳定的镜像源。

- **执行安装脚本**：
- `/bin/bash -c "$(curl -fsSL ``https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)``"`
- 跟着这里的脚本的提示一步一步来就行了
- **关键配置选择**：

  - **本体下载源**：选择 `2`（Gitee），避开了 `raw.githubusercontent.com` 的连接报错。
  - **软件镜像源**：选择 `5`（阿里巴巴），确保后续安装软件的速度。
- **激活环境变量**：
- 执行 `source ~/.zprofile` 将 `brew` 命令添加到系统搜索路径。
- **验证安装**：
- `brew -v`。

![](/static/Gv1mbDGwYoeK12xRp3gcdCeCnbg.png)

![](/static/MyMDbJvm7oIkVZxmmWac2YmMnyb.png)

![](/static/AGd7bs4yNoO39HxGTnvcD39inLE.png)

### 全局语言层：Python 3.10（我这边选择的是一个比较稳定的 python 版本）

虽然 macOS 自带 Python，但为了稳定性和后续开发，建议通过 Homebrew 安装一个独立的全局版本。

#### 2.1 安装命令

Bash

```
brew install python@3.10
```

---

#### 2.2 路径说明

- **安装位置**：`/opt/homebrew/bin/python3`（可以用 which python3.10 来看在哪）
- **用途**：用于运行简单的 Python 脚本、安装通用的工具包。

### 实验室管理层：Miniconda（比较方便一点）

对于 AI、机器学习或复杂项目，Miniconda 是实现“环境隔离”的核心工具。

#### 3.1 安装 Miniconda

Bash

```
brew install --cask miniconda
```

#### 3.2 初始化与激活

安装后需初始化 Shell，使 `conda` 命令生效：

Bash

```
/opt/homebrew/bin/conda init zsh
source ~/.zshrc
```

_现象_：终端提示符前出现 `(base)`，代表进入了 Conda 基础环境。

#### 3.3 配置国内镜像源（Conda）

为了加速第三方库的下载，配置清华镜像源：

Bash

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

---

### 编辑器配置：VS Code

去 [VS Code 官网](https://code.visualstudio.com/)。正常下载 dmg，然后安装就可以了。

![](/static/Etbkb7YngoWI3NxJDSJcNatqnPh.png)

---

### 如何开始写代码？（此部分都是跟 windows 那边通用的）

#### 步骤 A：创建独立虚拟环境（这部分建议自己学习一下 conda 环境相关的命令，网上很容易查得到，建议动手试一下）

永远不要在 `base` 环境下装库。为新项目创建一个新环境：

Bash

```
conda create -n my_project python=3.10
conda activate my_project
```

#### 步骤 B：在 VS Code 中关联环境

1. 在编辑器中打开项目文件夹。
2. 按 `Cmd + Shift + P` -> 输入 `Python: Select Interpreter`。
3. **选择你的 Conda 环境**（通常会标注为 `'my_project': conda`）。

---

### 维护常用命令速查

---

**配置提示**：

- **M 系列芯片加速**：安装 PyTorch 后，代码中可以使用 `device = torch.device("mps")` 来调用 Mac 的 GPU 进行加速。
- **保持整洁**：所有的 Conda 环境现在都规范地保存在 `/opt/homebrew/Caskroom/miniconda` 目录下，不再污染系统路径。

## Linux
