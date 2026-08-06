---
title: "Python 包管理"
order: 5
---

# Python 包管理

> ~~技术的本质就是调包~~~~！~~

「复用」是软件工程中一个十分重要的概念。如果我们想要实现的功能在项目的其他地方已经存在，那么我们就不必再次实现它。我们可以将程序的各个部分共享的逻辑提取出来，这样一来既可以避免代码的冗余，修改时也可以仅在一处修改，而不是在所有相同逻辑出现的地方进行修改。许多编程语言为了支持这样的复用都发展出了相应的机制，例如函数允许实现对于过程的复用，而语言中的类和对象和相关的 OOP 机制允许实现更加复杂的，对于数据结构和过程的复用。

然而，以上所说的这些复用机制仅仅局限于单个项目内。实际开发中，我们经常需要跨项目或跨团队共享和复用已有的功能。这就引出了「跨项目复用」的概念。我们可以把共同的逻辑提取出来，让这些逻辑本身成为一个单独的项目进行维护。在今天，你已经知道，这样的项目叫做「库」。今天的数字基础设施正是由一个又一个或大或小的「库」组成。得益于计算机科学的发展和自由软件运动，我们在今天可以轻而易举地获取到别人发布的开源库，并用在自己的项目中。这样一来，复用的规模被再一次扩大了，它并非局限于某个组织、国家或地区，而是所有人类程序员之间的复用。~~（为什么一定要是人类？）~~

![](/static/ESBAbbAr5o4mn1xFWq0c2JzWnuh.png)

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

## 虚拟环境

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

## Conda

在 AI 领域，很多包（如 `PyTorch`，`TensorFlow`）不仅包含 Python 代码，还深度依赖 C++ 库、CUDA 驱动等。`venv` 管不到这些非 Python 的二进制文件。我们需要比 `venv` 更加强大的虚拟环境管理。

Conda 管理的是**整个运行环境**，包括 Python 解释器本身、CUDA 工具链、C++ 编译器等。Conda 的环境通常是**全局集中管理**的。它会在你的电脑某个角落（如 `~/miniconda3/envs/`）开辟一个完整的隔离区。如果说 `venv` 靠的是修改 `sys.path` 这个 python 的包搜索路径，那么 Conda 更进一步，它还会修改 `PATH` 这样系统级的搜索路径。

Conda 是一个包管理的引擎，本体其实只有几十 MB 而已，有人围绕着它加入了各种预装的包，然后全部放在一起安装，形成了各种发行版。其中最有名，应用得也最广泛的是 Anaconda，它预装了 `NumPy`、`Pandas`、`Matplotlib`、`Scikit-learn` 等几乎所有数据科学必备的库，还提供了 Anaconda Navigator 这种图形界面，让你不用敲命令就能管理环境。在 Windows 上，很多 Python 库（比如带有 C++ 扩展的库）直接用 `pip` 装经常报错，Anaconda 预编译好了二进制文件，保证能在 Windows 上跑通。理所当然地，Anaconda 安装完的体积膨胀到了 3~5GB。

相比之下，Miniconda 提供了一个最小化的 conda 发行版，体积只有大概 300-500M，它足以满足几乎所有的日常使用。你可以到这里下载 Miniconda：[https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)。

安装完成之后，直接打开终端并输入 `conda`：

```powershell
conda
```

如果你是在 windows 上使用 conda，那么很遗憾，这个命令因为找不到 `conda` 而报错了。conda 的机制主要是通过管理环境变量来实现的，而 Windows 的环境变量管理和 Linux 不太一样，出于隔离环境变量的要求（你也不想 conda 环境里的环境变量搞坏了外面的东西），想要运行 conda，你要使用一个单独的命令提示符：

![](/static/BQ1Bbg8Ago72k9x9x2sc5og7nqf.png)

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

**查询环境列表**：`conda env list` 或 `conda info --envs`。它会列出你电脑上所有的工作区路径，当前激活的环境前面会标有一个星号。

**创建新环境**：`conda create -n <环境名> python=3.10`。建议养成在创建时显式指定 `python` 版本的习惯。如果不指定，Conda 可能会默认给一个不符合你项目要求的版本。

**激活与退出**：

- 进入环境：`conda activate <环境名>`
- 回到 base 环境：`conda deactivate`

**克隆环境**：`conda create -n <新名字> --clone <旧名字>`。

**删除环境**：`conda remove -n <环境名> --all`。

在激活了特定环境后，你就可以开始安装工具了。

**安装包**：`conda install <包名>`。如果需要特定版本，可以使用 `conda install <包名>=1.2.3`。Conda 会自动帮你分析依赖冲突。

**更新与卸载**：

- 更新：`conda update <包名>`
- 卸载：`conda remove <包名>`

**导出与复现**：当你完成了一个项目，需要把它交给同学时，使用：`conda env export > environment.yml` 对方只需要运行 `conda env create -f environment.yml`，就能还原一个一模一样的环境。

## uv

[uv](https://docs.astral.sh/uv/) 是一个使用 rust 编写的 python 包管理器。它的一大特点是**快**。在上一节里我们介绍了 conda，如果你尝试过使用 conda 安装包，你应该体会过 conda 在解析包依赖的这个阶段运行得十分缓慢。得益于 Rust 的底层性能，uv 的依赖解析和安装速度通常比传统工具快数十倍甚至上百倍。

同时，uv 借鉴了 Rust 的 Cargo 和前端工具的设计，原生支持基于 pyproject.toml 的工作流。以往在 conda 当中，我们往往需要导出 `environment.yml`，而在 uv 的包管理模式中，所有的依赖和它们的版本要求都被清晰地写入 pyproject.toml。在 pyproject.toml 中，我们通常只声明项目的顶层依赖和较为宽泛的版本要求。而 uv 在解析这些依赖后，会生成一个极其严谨的 uv.lock 锁文件。这个文件记录了整个依赖树中每一个包的精确版本号甚至哈希值。只要基于同一个锁文件构建环境，安装的依赖分支将完全一致，这就从根本上保证了项目依赖的可复现性。

在传统的 pip 或 conda 工作流中，如果你在十个不同的项目里都用到了同一个版本的重型依赖，比如 PyTorch 或是 Transformers，你的硬盘上就会真的存下十份一模一样的庞大文件。而 uv 彻底改变了这一点。当你第一次下载某个包时，uv 会把它存放在系统的全局缓存目录中。之后无论你在多少个新的虚拟环境或是项目中需要安装这个特定版本的包，uv 会直接利用操作系统的硬链接机制，将虚拟环境中的包指向全局缓存里的同一份文件，如果你熟悉一些前端工具，你会发现这和 pnpm 是非常类似的。

由于 uv 是一个用 Rust 编译出的独立二进制文件，它的安装和运行完全不需要依赖系统中现有的 Python 环境。

在 Linux 或 macOS 系统中，你可以直接通过终端一键安装：

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果你使用 windows，也可以：

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
