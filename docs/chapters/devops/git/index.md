---
title: "Git 与 GitHub"
order: 4
---
# Git 与 GitHub

> [!TIP]
> 在阅读此文前，建议先阅读 [Visual Studio Code](/chapters/devops/vscode.html) 和 [终端与命令行](/chapters/devops/terminal.html)。因为安装 Git Bash 时，可以设置 VS Code 作为默认编辑器。且使用 Git Bash 需要读者有初步的命令行知识。

> [!IMPORTANT]
> 本文的操作全部基于 Windows 10/11。命令行的运行全部使用 Git Bash。

## 为什么需要 Git？

如果你写过一些需要频繁改动的文档，你一定不会对这样的文件命名感到陌生：

> `项目_草稿.docx`
> 
> `项目_修改版.docx`
> 
> `项目_最终版.docx`
> 
> `项目_绝对不改最终版.docx`
> 
> `项目_打死不改最终版_v2.docx`

当你发现 `绝对不改版` 里有一段内容写坏了，想要找回第一天写的 `草稿` 里的某句话时，往往已经找不到了。如果这时候还要和三四个同学一起改同一份文件，大家互相传来传去，最后根本不知道谁改了什么，甚至会把别人的心血不小心覆盖掉。

Git 就是为了终结这种混乱而诞生的。 简单来说，Git 是一个版本控制系统。你只需要把每次更改的内容告诉 Git，它就能帮你自动、高效地管理文件的每一次修改。既方便、又不会因为每个版本都存成一个文件而大大占用磁盘空间。

## 安装 Git

1. 进入[官网](https://git-scm.com/install/windows)，选择 “Git for Windows/x64 Setup” 下载。下载可能较慢，读者可以参考 [爱的魔法](/chapters/devops/vpn.html) 加快下载速度。
   ![](install_page.png)
2. 一路点下一步，到“Choosing the default editor”这一步时，推荐选择“[Visual Studio Code](/chapters/devops/vscode.html)”。
   ![](git_choose_editor.png)

> [!TIP]
> 作为 Git 的默认编辑器，Vim 是一个强大、历史悠久、轻量且插件丰富的终端编辑器，但其不直观的键位绑定和晦涩的操作方式不适合新手。我们建议你使用具有现代 GUI 的 Visual Studio Code。

3. 一路点下一步，直到安装完成。此时，右键文件管理器中任一文件夹的空白部分，应该会出现“Open Git Bash here”选项（Windows 11 用户需要点击“显示更多选项”来发现这个选项）。
   ![](open_git_bash_here.png)
4. 点击“Open Git Bash here”。如果出现一个如下图所示的黑窗口，证明 Git 安装成功。
   ![](git_bash.png)

> [!Tip] 什么是 Git Bash？
> Git Bash 是 Windows 系统上的类 Unix 命令行环境。它不仅内置了完整的 Git 版本控制工具，还支持常用的 Linux 命令行语法，是 Windows 开发者在终端中高效执行代码管理与 Shell 脚本的首选工具之一。有关命令行的相关知识，参见 [终端与命令行](/chapters/devops/terminal.html)。

## 使用 Git

### 创建仓库

仓库（Repository，简称 Repo）就是 Git 用来保存你所有文件历史记录的“小基地”。你可以把任何一个普通文件夹变成 Git 仓库。接下来让我引导你一步步创建第一个 Git 仓库。

1. 选择一个合适的地方，新建一个空文件夹，这就是我们的仓库所在的文件夹；
2. 在这个空文件夹右键，选择 “Open Git Bash here”，打开 Git Bash。
3. 输入以下命令：

```sh
git init
```

  如果你是第一次使用 Git，那么 Git 大概会输出以下信息：

```
hint: Using 'master' as the name for the initial branch. This default branch name
hint: will change to "main" in Git 3.0. To configure the initial branch name
hint: to use in all of your new repositories, which will suppress this warning,
hint: call:
hint:
hint:   git config --global init.defaultBranch <name>
hint:
hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
hint: 'development'. The just-created branch can be renamed via this command:
hint:
hint:   git branch -m <name>
hint:
hint: Disable this message with "git config set advice.defaultBranchName false"
Initialized empty Git repository in /home/git-test-user/git/.git/
```

  上面的输出告诉我们：
```
Initialized empty Git repository in /home/git-test-user/git/.git/
```
  这说明当前目录已经顺利初始化，成为了一个 Git 仓库！此时文件夹下会生成一个隐藏的 `.git` 文件夹，这就是 Git 用来记录版本历史的核心数据库，请不要手动修改或删除它。

## 第一次提交变更

在动手敲命令之前，我们需要先搞懂 Git 里的两个核心概念：**暂存区（Staging Area）** 和 **提交（Commit）**。

我们可以把 Git 的工作方式类比为**“拍照发朋友圈”**：

* **工作区（Working Directory）**：就是你当前的文件夹。你在里面新建、修改或删除文件的过程，就像是在**布景和摆姿势**。
* **暂存区（Staging Area）**：就像是**打开相机胶卷，选择要上传的照片**。你可以选一张，也可以选多张。选中的文件就是被 `git add` 标记的文件。
* **提交（Commit）**：相当于**点击“发送朋友圈”**。一旦点击提交，这一刻文件夹的状态就会被永久按下快门，生成一个独一无二的“历史记录版本”。

也就是说：**修改文件（准备） $\rightarrow$ `git add`（挑选） $\rightarrow$ `git commit`（拍照保存）**。

了解了这个流程，我们来动手操作一遍：

### 新建测试文件

在当前文件夹下创建一个名为 `README.md` 的文本文件，并在里面写上一句：`Hello World!`。

### 查看仓库状态
想知道现在文件夹里有什么变化，可以随时运行：

```sh
git status
```

输出：
```
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        README.md

nothing added to commit but untracked files present (use "git add" to track)
```

你会看到 `README.md` 显示为红色（Untracked files），表示 Git 注意到了这个新文件，但它目前还是草稿，尚未被放入暂存区。

### 将文件添加到暂存区
使用以下命令把 README.md 放到暂存区：

```sh
git add README.md
```
此时再次运行 `git status`：

```
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   README.md
```

你会发现文件名变成了绿色（Changes to be committed），说明它已经准备好提交了。

> [!TIP]
> 如果修改了多个文件，可以直接运行 `git add .`（注意后面有一个英文句号），它会将当前目录下所有有变化的文件一次性全部放入暂存区。

### 提交变更到本地仓库 (git commit)
使用 `git commit` 命令正式把暂存区里的变更永久保存下来。为了以后能看懂这次提交改了什么，我们需要用 `-m` 参数附上一句简单明确的提交说明：

```sh
git commit -m "feat: 新建 README 文件"
```

按下回车。如果你第一次使用 Git，大概率会有下面的提示：

```
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'git-test-user@sakimidare-arch.(none)')
```

这是因为为了高效地管理版本，在第一次使用 Git 前，Git 需要知道你的名字和邮箱，这些信息会附加在你提交的代码记录中：

使用

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

来告诉 Git 你是谁，然后重新运行：

```sh
git commit -m "feat: 新建 README 文件"
```

出现这些信息：
```
[master (root-commit) 6b83eb4] feat: 新建 README 文件
 1 file changed, 1 insertion(+)
 create mode 100644 README.md
```

就代表我们的第一个提交创建成功了！

### 查看提交历史

运行
```sh
git log
```
你会看到一条包含你的名字、邮箱、提交时间和提交说明的清晰记录。

```
commit 6b83eb430a3d1e3e1dcf02ef7f006b0deedad6d6 (HEAD -> master)
Author: SakiMidare <sakimidare@outlook.com>
Date:   Sat Aug 8 13:54:22 2026 +0800

    feat: 新建 README 文件
```

> [!TIP] 什么是 Commit Hash？
> 运行 git log 时，每条记录开头那串由字母和数字组成的长字符串（如 上面的`6b83eb4...`）叫做 Commit Hash。它是 Git 根据本次提交的所有内容（作者、时间、改动文件等）算出的唯一身份 ID。后续如果想要回退版本、查看某次修改，只需要提供这个 Hash 的前 7 位字母数字即可。


## 什么是 GitHub？

如果说 **Git** 是你电脑里的“本地相册”，那么 **GitHub** 就是“代码界的朋友圈”。

GitHub 是目前全球最大的开源代码托管平台，基于 Git 构筑。通过 GitHub，你可以：
1. **云端备份**：把本地代码同步到云端，换台电脑也能继续工作，再也不用担心硬盘损坏。
2. **开源交流**：浏览、学习、使用全球优秀开发者开源的软件和代码。
3. **多人协作**：和团队成员在同一个项目中并行开发、提交代码、互评审查。

> [!NOTE] 区分 Git 与 GitHub
> * **Git**：一个**本地命令行软件**（工具），用来记录文件的版本历史。
> * **GitHub**：一个**在线网站平台**（服务），用来在云端保存、展示和分享 Git 仓库。

## 将代码推送到 GitHub

接下来，我们把刚才在本地创建并提交的仓库，同步到 GitHub 云端。

### 注册与登录 GitHub

1. 打开 [GitHub](https://github.com/)。
2. 点击右上角 **Sign up** 注册账号（如果已有账号直接点击 **Sign in** 登录）。
3. 按照提示完成邮箱验证。

> [!TIP]
> 如果遇到 GitHub 网页加载缓慢或无法打开的情况，可以再次借助 [爱的魔法](/chapters/devops/vpn.html)。

### 将本地 SSH 公钥上传到 GitHub 上

> [!TIP]
> 建议阅读：[使用 SSH 连接到GitHub](https://docs.github.com/zh/authentication/connecting-to-github-with-ssh)

我们可以把 SSH 密钥想象成一对密码锁：

* **私钥（Private Key）**：保存在你本地电脑上，绝对不能给别人看。
* **公钥（Public Key）**：上传到你的 GitHub 账号里，可以公开。

只要两边对上了，以后你在这台电脑上推拉代码时，GitHub 就能自动确认你的身份，**再也不需要每次都输入密码或进行网页验证**。

创建 SSH 密钥对的步骤请参考 [SSH 远程登录](/chapters/devops/ssh.html)，如果已创建密钥对，则忽略这一步。

创建完成之后，请查看你的公钥：

```sh
cat ~/.ssh/id_ed25519.pub
```

复制完整输出内容（一般是一个字符串和一个邮箱），粘贴到 GitHub 设置 - SSH Key - Add new SSH Key 的 Key 输入框中，标题可以随便取名（见下）：

![](github_ssh_key_setup.png)

点击 “Add SSH Key” 并验证通过后。

如果配置完成后，运行

```sh
ssh -T git@github.com
```

输出

```
Hi <用户名>! You've successfully authenticated, but GitHub does not provide shell access.

```

而不是

```
git@github.com: Permission denied (publickey).
```

证明我们的 SSH Key 配置成功，可以用这个 Key 来访问 GitHub 上的仓库了！

> [!TIP]
> 如果你遇到这种输出：
> ```
> ED25519 key fingerprint is: SHA256:+xxxxx
> This key is not known by any other names.
> Are you sure you want to continue connecting (yes/no/[fingerprint])?
> ```
> 说明你的电脑没有信任 GitHub 的 公钥。请手动输入 `yes`，信任 GitHub 的公钥。

### 在 GitHub 上新建远程仓库

1. 登录后，点击页面右上角的 **`+`** 号，选择 **New repository**。

  ![](github_plus.png)

2. 填写仓库信息：
   * **Repository name（仓库名称）**：填入与本地文件夹一致的名称（例如 `first-repo`）。
   * **Description（可选描述）**：简单写一句项目介绍。
   * **Public / Private（公开/私有）**：初学者建议选择 **Public**（所有人可见）。
   * **不要** 勾选 *Add a README file*、*.gitignore* 或 *Choose a license*（因为我们本地已经创建过文件了，保持远端完全空白）。

3. 点击底部的绿色按钮 **Create repository**。

![](github_create_repo.png)

### 连接本地仓库与远程仓库

创建成功后，GitHub 会展示一个页面，其中包含了关键的命令指南：

  ![](github_quick_setup.png)

我们想要把本地的仓库推送到 GitHub 上，请在 Git Bash 中依次进行：

#### 关联远程仓库地址

> [!CAUTION]
> GitHub 出于安全考虑，**已不再允许直接在终端输入账户密码进行登录**！
> ```
> git remote add origin https://github.com/你的用户名/你的仓库名.git
> ```
> 在推送时很可能出现
> ```
> remote: Invalid username or token. Password authentication is not supported for Git operations.
> fatal: Authentication failed for 'https://github.com/你的用户名/你的仓库名.git/'
> ```
> 的报错。

因为我们已经配置好了 SSH，所以我们将 HTTP 协议换成 SSH 协议。执行

```sh
git remote add origin git@github.com:你的用户名/你的仓库名.git
```

添加远程仓库地址。

#### 重命名默认分支为 `main`

```sh
git branch -M main
```

> [!NOTE] 为什么需要这一步？
> 过去 Git 的默认分支名称是 master，而 GitHub 目前默认采用 main 作为主分支名。这条命令可以将本地的 master 分支重命名为 main，以保持一致。

#### 推送至远程
```sh
git push -u origin main
```
如果输出：

```
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Writing objects: 100% (3/3), 248 bytes | 248.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
To github.com:用户名/仓库.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

说明你的第一个仓库已经推送到了 GitHub 上了！

![](github_demo.png)