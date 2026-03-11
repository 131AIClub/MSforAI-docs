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
