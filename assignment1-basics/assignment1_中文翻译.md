# CS336 作业 1（基础）：构建一个 Transformer 语言模型

版本 26.0.3
CS336 课程组
2026 年春季

---

## 1 作业概述

在这份作业中，你将构建训练一个标准 Transformer 语言模型（LM）所需的全部组件，并训练一些模型。

**你将实现的内容**
1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数和 AdamW 优化器（第 4 节）
4. 训练循环，支持模型和优化器状态的序列化与加载（第 5 节）

**你将运行的内容**
1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 用训练好的分词器对数据集进行编码，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer 语言模型。
4. 使用训练好的 Transformer LM 进行样本生成和困惑度评估。
5. 在 OpenWebText 上训练模型，并将你达到的困惑度提交到排行榜。

**你可以使用的内容**
我们期望你**从零**构建每个组件。特别是，你不可以使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的以下内容之外的任何定义：
- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）¹
- `torch.optim.Optimizer` 基类

你可以使用任何其他 PyTorch 定义。如果你想使用某个函数或类但不确定是否允许，请随时在 Slack 上提问。拿不准时，请考虑使用它是否会损害本作业的"从零实现"精神。

**关于 AI 工具的声明**
AI 可以完全自主地解决作业的很多部分。这使得深入理解并学习课程材料变得更加困难。

允许使用 AI 工具来回答高层的概念性问题，或提供底层的编程文档（如函数签名和库 API）。但是，不允许使用 AI 工具实现本作业的任何部分。这既包括编码代理（例如 Cursor Agents、Codex、Claude Code），也包括 AI 自动补全（例如 Cursor Tab、GitHub Copilot）。使用 AI 代理时，请确保它使用所提供的 `AGENTS.md` 文件。使用聊天机器人时也应包含提示词。

¹ 完整列表请参见 pytorch.org/docs/stable/nn.html#containers。

我们强烈建议你在完成作业时在你的 IDE 中禁用 AI 自动补全（例如 Cursor Tab、GitHub Copilot）（尽管非 AI 自动补全，例如自动补全函数名，是完全没问题的）。往年学生强调，禁用 AI 自动补全让他们更容易深入学习材料。

完整 AI 政策，请参见此文档。

**代码结构**
作业代码以及本说明文档都可以在 GitHub 上获取：
github.com/stanford-cs336/assignment1-basics

请 `git clone` 该仓库。如果有更新，我们会通知你，以便你 `git pull` 获取最新版本。

1. `cs336_basics/*`：这是你写代码的地方。注意这里**没有**代码——你可以完全从头开始做任何事！
2. `adapters.py`：有一组你的代码必须拥有的功能。对于每项功能（例如缩放点积注意力），只需通过调用你的代码来填充其实现（例如 `run_scaled_dot_product_attention`）。注意：你对 `adapters.py` 的修改不应包含任何实质性逻辑；这是胶水代码。
3. `test_*.py`：这里包含你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），它们会调用 `adapters.py` 中定义的钩子。不要编辑测试文件。

**如何提交**
为了提交，运行 `make_submission.sh` 来构建一个提交用的 zip 文件。如果你有大文件的数据或你不想包含在提交 zip 中的检查点，请确保将额外的文件添加到脚本的排除列表中。

你将向 Gradescope 提交以下文件：
- `writeup.pdf`：回答所有书面问题。请用排版工具作答。
- `code.zip`：包含你写的所有代码。

要向排行榜提交，请向以下地址提交 PR：
github.com/stanford-cs336/assignment1-basics-leaderboard

有关排行榜仓库的详细提交说明，请参见其中的 README.md。

**从哪里获取数据集**
本作业将使用两个预处理过的数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。两个数据集都是单个的大型纯文本文件。

如果你是在上课期间做这份作业，你可以在计算指南中找到下载数据集的说明。
如果你是自行学习，你可以用 README.md 中的命令下载这些文件。

**低资源提示：初始化**
在整门课程的作业说明中，我们会给出如何在较少或没有 GPU 资源的情况下完成作业的建议。例如，我们有时会建议缩小数据集或模型规模，或解释如何在 Mac 集成 GPU 或 CPU 上运行训练代码。你会在蓝色框（像这个）中找到这些"低资源提示"。即使你是能够使用课程机器的注册斯坦福学生，这些提示也可能帮助你更快地迭代并节省时间，所以我们建议阅读它们！

**低资源提示：在 Apple Silicon 或 CPU 上完成作业 1**
使用工作人员解答代码，我们可以在 Apple M4 Max 芯片（36 GB 内存）上训练一个能生成相当流畅文本的语言模型，在 Metal GPU（MPS）上不到 5 分钟，使用 CPU 约 30 分钟。如果这些词对你不算什么，别担心！只要知道，如果你有一台还算新的笔记本电脑，并且你的实现正确且高效，你就能训练出一个小型 LM，生成相当流畅的简单儿童故事。
在本作业稍后部分，我们会解释如果你使用 CPU 或 MPS 需要做哪些调整。

---

## 2 字节对编码（BPE）分词器

在本作业的第一部分，我们将训练并实现一个字节级别的字节对编码（BPE）分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。具体来说，我们将任意（Unicode）字符串表示为字节序列，并在该字节序列上训练我们的 BPE 分词器。稍后，我们将使用这个分词器将文本（字符串）编码为用于语言建模的标记（token，整数序列）。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，它将字符映射为整数码点（code point）。截至 Unicode 17.0（2025 年 9 月发布），该标准在 172 种书写系统中定义了 159,801 个字符。例如，字符 `s` 的码点是 115（通常记为 U+0073，其中 U+ 是惯例前缀，0073 是 115 的十六进制表示），而字符 `牛` 的码点是 29275。在 Python 中，你可以使用 `ord()` 函数将单个 Unicode 字符转换为它的整数表示。`chr()` 函数将整数 Unicode 码点转换为包含相应字符的字符串。

```
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

**问题（unicode1）：理解 Unicode（1 分）**
(a) `chr(0)` 返回哪个 Unicode 字符？
交付物：一句话回答。
(b) 这个字符的字符串表示（`__repr__()`）与其打印表示有何不同？
交付物：一句话回答。
(c) 当这个字符出现在文本中时会发生什么？在你的 Python 解释器中玩玩下面这些内容，看看是否符合你的预期，这可能会有所帮助：
```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
交付物：一句话回答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不切实际的，因为词汇表会大到难以处理（大约 15 万个条目），而且非常稀疏（因为很多字符相当罕见）。作为替代，我们将使用一种 Unicode 编码，它将 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主导编码（超过 98% 的网页）。

要将 Unicode 字符串编码为 UTF-8，我们可以使用 Python 中的 `encode()` 函数。要访问 Python `bytes` 对象底层的字节值，我们可以遍历它（例如调用 `list()`）。最后，我们可以使用 `decode()` 函数将 UTF-8 字节字符串解码为 Unicode 字符串。

```
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码后字符串的字节值（0 到 255 的整数）。
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将我们的 Unicode 码点转换为字节序列（例如通过 UTF-8 编码），我们实质上是在将一个码点序列（21 位整数，有 159,801 个有效值）转换为字节值序列（0 到 255 的整数）。长度为 256 的字节词汇表处理起来要容易得多。使用字节级别分词时，我们无需担心词汇表外的标记，因为我们知道任何输入文本都可以表示为 0 到 255 的整数序列。

**问题（unicode2）：Unicode 编码（3 分）**
(a) 相对于 UTF-16 或 UTF-32，有哪些理由更倾向于在 UTF-8 编码的字节上训练我们的分词器？对各种输入字符串比较这些编码的输出可能会有所帮助。
交付物：一到两句话的回答。
(b) 考虑下面这个（错误的）函数，它本意是将 UTF-8 字节字符串解码为 Unicode 字符串。为什么这个函数是错误的？给出一个会产生错误结果的输入字节字符串示例。
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
交付物：一个输入字节字符串示例，使得 `decode_utf8_bytes_to_str_wrong` 产生错误输出，并附一句话说明该函数错误的原因。
(c) 给出一个不能解码为任何 Unicode 字符的两字节序列。
交付物：一个示例，并附一句话说明。

### 2.3 子词分词

虽然字节级别分词可以缓解词级别分词器面临的词汇表外问题，但将文本分词为字节会导致极长的输入序列。这会减慢模型训练，因为一个有 10 个单词的句子在词级别语言模型中可能只有 10 个标记，但在字符级模型中可能有 50 个甚至更多的标记（取决于单词的长度）。处理这些更长的序列需要每个步骤更多的计算。此外，在字节序列上进行语言建模很困难，因为更长的输入序列会引入数据中的长期依赖。

子词分词是词级别分词器和字节级别分词器之间的一种折中方案。注意，字节级别分词器的词汇表有 256 个条目（字节值为 0 到 255）。子词分词器用更大的词汇表来换取输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在我们的原始文本训练数据中经常出现，那么给它分配一个词汇表条目就会把这个 3 标记序列减少为单个标记。

我们如何选择要添加到词汇表中的这些子词单元？R. Sennrich 等人 [3] 提出使用字节对编码（BPE；P. Gage [5]），这是一种压缩算法，它迭代地将最频繁的字节对"替换"（"合并"）为一个新的、尚未使用的索引。注意，该算法向我们的词汇表添加子词标记，是为了最大化我们输入序列的压缩——如果一个词在我们的输入文本中出现的次数足够多，它就会被表示为单个子词单元。

通过 BPE 构建词汇表的子词分词器通常被称为 BPE 分词器。在本作业中，我们将实现一个字节级别的 BPE 分词器，其中词汇表项是字节或字节的合并序列，这使我们在词汇表外处理和输入序列长度可控方面兼得两全。构建 BPE 分词器词汇表的过程被称为"训练"BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器训练过程包括三个主要步骤。

**词汇表初始化**
分词器词汇表是从字节串标记到整数 ID 的一一映射。由于我们训练的是字节级别的 BPE 分词器，我们的初始词汇表就是所有字节的集合。因为有 256 种可能的字节值，我们的初始词汇表大小为 256。

**预分词**
一旦有了词汇表，原则上你可以统计文本中字节相邻出现的频率，并从最频繁的字节对开始合并。然而，这在计算上相当昂贵，因为每次合并我们都要对整个语料库进行完整遍历。此外，直接在语料库上合并字节可能会导致只差在标点符号上的标记（例如 `dog!` 对比 `dog.`）。这些标记会得到完全不同的标记 ID，尽管它们可能具有很高的语义相似性（因为它们只差在标点符号上）。

为避免这种情况，我们对语料库进行预分词。你可以把这看作语料库上的一个粗粒度分词，帮助我们统计字符对出现的频率。例如，单词 `text` 可能是一个出现 10 次的预标记。在这种情况下，当我们统计字符 `t` 和 `e` 相邻出现的频率时，我们会看到单词 `text` 中 `t` 和 `e` 相邻，因此我们可以将它们的计数增加 10，而不是遍历整个语料库。由于我们在训练字节级别的 BPE 模型，每个预标记都表示为 UTF-8 字节序列。

R. Sennrich 等人 [3] 的原始 BPE 实现通过简单地在空白处分割来进行预分词（即 `s.split(" ")`）。这种方法在基于 SentencePiece 的分词器中仍然可以发现（例如 Llama 1 和 2 的分词器）。

大多数现代分词器使用基于正则的预分词器，这是 GPT-2 的做法；A. Radford 等人 [6]。我们将使用原始正则的一个略微好看的形式，取自 github.com/openai/tiktoken/pull/234/files：

```
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地用这个预分词器切分一些文本以获得对其行为的更好感受可能会有帮助：
```
>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

然而，在你代码中使用它时，你应该使用 `re.finditer`，以避免在构建从预标记到其计数的映射时存储预分词后的单词。

**计算 BPE 合并**
现在我们已经将输入文本转换为预标记，并将每个预标记表示为 UTF-8 字节序列，我们可以计算 BPE 合并（即训练 BPE 分词器）。从高层看，BPE 算法迭代地统计每一对字节，并识别频率最高的那一对（"A"、"B"）。最频繁对（"A"、"B"）的每次出现随后被合并，即被替换为新的标记 "AB"。这个新的合并标记被添加到我们的词汇表中；因此，BPE 训练后的最终词汇表大小等于初始词汇表大小（我们的情况是 256）加上训练期间执行的 BPE 合并操作次数。为了 BPE 训练的效率，我们不考虑跨越预标记边界的对。² 在计算合并时，以确定性方式打破对频率的平局：优先选择字典序更大的对。例如，如果对 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 都有最高频率，我们会合并 ("BA", "A")：
```
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

² 注意，R. Sennrich 等人 [3] 的原始 BPE 公式规定了包含一个词尾标记。我们在训练字节级别 BPE 模型时不添加词尾标记，因为所有字节（包括空白和标点符号）都包含在模型的词汇表中。由于我们明确表示空格和标点，学习到的 BPE 合并自然会反映这些词边界。

**特殊标记**
通常，一些字符串（例如 `<|endoftext|>`）用于编码元数据（例如文档之间的边界）。在编码文本时，通常希望将某些字符串视为"特殊标记"，它们永远不应被拆分为多个标记（即始终保留为单个标记）。例如，序列结束字符串 `<|endoftext|>` 应始终保留为单个标记（即单个整数 ID），这样我们就知道何时停止从语言模型生成。这些特殊标记必须被添加到词汇表中，以便它们有对应的固定标记 ID。

R. Sennrich 等人 [3] 的算法 1 包含一个低效的 BPE 分词器训练实现（基本上遵循我们上面概述的步骤）。作为第一个练习，实现并测试这个函数以检查你的理解可能会有所帮助。

**示例（bpe_example）：BPE 训练示例**
这里有一个来自 R. Sennrich 等人 [3] 的程式化示例。考虑一个由以下文本组成的语料库：
```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```
并且词汇表有一个特殊标记 `<|endoftext|>`。

**词汇表**
我们用特殊标记 `<|endoftext|>` 和 256 个字节值来初始化词汇表。

**预分词**
为简单起见并聚焦于合并过程，我们在这个示例中假设预分词简单地按空白分割。当我们预分词并计数时，我们得到频率表：
```
{low: 5, lower: 2, widest: 3, newest: 6}
```
将其表示为 `dict[tuple[bytes, ...], int]` 很方便，例如 `{(l,o,w): 5, …}`。注意，即使是单个字节在 Python 中也是 `bytes` 对象。Python 中没有表示单个字节的 `byte` 类型，就像没有表示单个字符的 `char` 类型一样。

**合并**
我们首先查看每一对连续的字节，并对它们出现的单词的频率求和 `{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。对 ('e', 's') 和 ('s', 't') 频率相同，因此我们取字典序更大的对 ('s', 't')。然后我们合并预标记，最终得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

在第二轮中，我们看到 (e, st) 是最常见的对（计数为 9），我们会合并为 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续这样，我们最终得到的合并序列将是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果我们取 6 次合并，我们得到 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词汇表元素将是 `[<|endoftext|>, [...256 字节字符], st, est, ow, low, west, ne]`。

有了这个词汇表和合并集合，单词 `newest` 将分词为 [ne, west]。

### 2.5 尝试 BPE 分词器训练

让我们在 TinyStories 数据集上训练一个字节级别的 BPE 分词器。找到/下载数据集的说明见第 1 节。在开始之前，我们建议先看一下 TinyStories 数据集，以了解数据里有什么。

**并行化预分词**
你会发现一个主要的瓶颈是预分词步骤。你可以通过使用内置库 `multiprocessing` 并行化你的代码来加速预分词。具体来说，我们建议在预分词的并行实现中，对语料库进行分块，同时确保你的分块边界出现在特殊标记的开头。你可以自由地原样使用以下链接的入门代码来获取分块边界，然后将其用于跨进程分配工作：
https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

这种分块总是有效的，因为我们绝不想跨越文档边界进行合并。为本作业的目的，你总是可以这样分割。不用担心接收到一个不包含 `<|endoftext|>` 的非常大的语料库的边界情况。

**预分词前移除特殊标记**
在运行使用正则模式（使用 `re.finditer`）的预分词之前，你应该从你的语料库（或分块，如果使用并行实现）中剥离所有特殊标记。确保你在特殊标记处分割，这样就不会跨越它们所界定的文本进行合并。例如，如果你有一个像 `[Doc 1]<|endoftext|>[Doc 2]` 的语料库（或分块），你应该在特殊标记 `<|endoftext|>` 处分割，并分别对 `[Doc 1]` 和 `[Doc 2]` 进行预分词，这样就不会跨越文档边界进行合并。换句话说，特殊标记在训练期间定义了硬分割边界，但它们本身不应计入合并统计。这可以使用以 `"|".join(special_tokens)` 作为分隔符的 `re.split` 来完成（由于 `|` 可能出现在特殊标记中，需要小心使用 `re.escape`）。测试 `test_train_bpe_special_tokens` 将对此进行测试。

**优化合并步骤**
上述程式化示例中 BPE 训练的朴素实现很慢，因为对于每次合并，它都遍历所有字节对以识别最频繁的对。然而，每次合并后只有与合并对重叠的对计数会改变。因此，BPE 训练速度可以通过索引所有对的计数并增量更新这些计数来改进，而不是显式遍历每对字节来统计对频率。这种缓存过程可以带来显著的加速，尽管我们注意到 BPE 训练的合并部分在 Python 中不可并行化。

**低资源提示：性能分析**
你应该使用 cProfile 或 py-spy 等性能分析工具来识别实现中的瓶颈，并专注于优化这些瓶颈。

**低资源提示："缩小规模"**
与其直接跳到在完整 TinyStories 数据集上训练你的分词器，我们建议你首先在数据的小子集上训练：一个"调试数据集"。例如，你可以改为在 TinyStories 验证集上训练你的分词器，它是 2.2 万个文档而不是 212 万个。这说明了在可能的情况下缩小规模以加速开发的一般策略：例如，使用较小的数据集、较小的模型大小等。选择调试数据集的大小或超参数配置需要仔细考虑：你希望你的调试集足够大，以便与完整配置具有相同的瓶颈（这样你做的优化可以推广），但又不能太大以至于运行时间过长。

**问题（train_bpe）：BPE 分词器训练（15 分）**
交付物：编写一个函数，给定输入文本文件的路径，训练一个（字节级别）BPE 分词器。你的 BPE 训练函数应处理（至少）以下输入参数：

输入：
- `input_path: str` — 包含 BPE 分词器训练数据的文本文件路径。
- `vocab_size: int` — 一个正整数，定义最大的最终词汇表大小（包括初始字节词汇表、合并产生的词汇表项以及任何特殊标记）。
- `special_tokens: list[str]` — 要添加到词汇表的字符串列表。训练期间，将它们视为防止跨其跨度合并的硬边界，但在计算合并统计时不包含它们。

你的 BPE 训练函数应返回生成的词汇表和合并：

输出：
- `vocab: dict[int, bytes]` — 分词器词汇表，从 int（词汇表中的标记 ID）到 bytes（标记字节）的映射。
- `merges: list[tuple[bytes, bytes]]` — 训练产生的 BPE 合并列表。每个列表项是一个字节元组（`<token1>`, `<token2>`），表示 `<token1>` 与 `<token2>` 合并了。合并应按创建顺序排列。

为了对照我们提供的测试测试你的 BPE 训练函数，你首先需要实现 [adapters.run_train_bpe] 处的测试适配器。然后，运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过所有测试。可选（这可能是一个很大的时间投入），你可以使用某种系统语言实现训练方法的关键部分，例如 C++（考虑 cppyy 或 nanobind）或 Rust（使用 PyO3）。如果你这样做，请注意哪些操作需要复制 vs 直接从 Python 内存读取，并确保留下构建说明，或确保它仅使用 pyproject.toml 就能构建。另请注意，GPT-2 正则不受大多数正则引擎的良好支持，而且在大多数支持它的引擎上会太慢。我们已经验证 Oniguruma 相当快且支持负向先行断言，但 Python 中的 regex 包甚至更快。

**问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）**
(a) 在 TinyStories 数据集上训练一个字节级别 BPE 分词器，最大词汇表大小为 10,000。确保将 TinyStories 的 `<|endoftext|>` 特殊标记添加到词汇表中。将生成的词汇表和合并序列化到磁盘以供进一步检查。训练花了多少时间和内存？词汇表中最长的标记是什么？它合理吗？
资源需求：≤ 30 分钟（无需 GPU），≤ 30 GB 内存
提示 你应该能够在预分词时使用多进程以及以下两个事实，在 2 分钟内完成 BPE 训练：
(a) `<|endoftext|>` 标记分隔数据文件中的文档。
(b) `<|endoftext|>` 标记在应用 BPE 合并之前作为特殊情况处理。
交付物：一到两句话的回答。
(b) 对你的代码进行性能分析。分词器训练过程的哪一部分占用的时间最多？
交付物：一到两句话的回答。

接下来，我们将尝试在 OpenWebText 数据集上训练一个字节级别 BPE 分词器。和之前一样，我们建议查看数据集以更好地了解其内容。

**问题（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）**
(a) 在 OpenWebText 数据集上训练一个字节级别 BPE 分词器，最大词汇表大小为 32,000。将生成的词汇表和合并序列化到磁盘以供进一步检查。词汇表中最长的标记是什么？它合理吗？
资源需求：≤ 12 小时（无需 GPU），≤ 100 GB 内存
交付物：一到两句话的回答。
(b) 比较和对比你在 TinyStories 与 OpenWebText 上训练得到的分词器。
交付物：一到两句话的回答。

### 2.6 BPE 分词器：编码与解码

在作业的前一部分中，我们实现了一个函数，用于在输入文本上训练 BPE 分词器，从而获得分词器词汇表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，它可以加载给定的词汇表和合并列表，并使用它们将文本编码为 token ID 以及将 token ID 解码回文本。

#### 2.6.1 编码文本

BPE 编码文本的过程与我们训练 BPE 词汇表的过程相镜像。主要有以下几个步骤。

**步骤 1：预分词（Pre-tokenize）。** 我们首先对序列进行预分词，并将每个预 token 表示为 UTF-8 字节序列，正如我们在 BPE 训练中所做的那样。我们将在每个预 token 内部把这些字节合并为词汇表元素，每个预 token 独立处理（不跨预 token 边界合并）。

**步骤 2：应用合并。** 然后，我们取出 BPE 训练期间创建的词汇表元素合并序列，并按创建时的相同顺序将其应用到我们的预 token 上。

**示例 (bpe_encoding)：BPE 编码示例**

例如，假设我们的输入字符串是 `'the cat ate'`，我们的词汇表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，学到的合并是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，我们的预分词器会将这个字符串拆分为 `['the', ' cat', ' ate']`。然后，我们将查看每个预 token 并应用 BPE 合并。

第一个预 token `'the'` 最初表示为 `[b't', b'h', b'e']`。查看我们的合并列表，我们发现第一个适用的合并是 `(b't', b'h')`，用它把预 token 转换为 `[b'th', b'e']`。然后，我们回到合并列表，发现下一个适用的合并是 `(b'th', b'e')`，它将预 token 转换为 `[b'the']`。最后，再次查看合并列表，我们发现没有更多适用于该字符串的合并（因为整个预 token 已经合并为单个 token），所以我们完成了 BPE 合并的应用。对应的整数序列是 `[9]`。

对剩余的预 token 重复此过程，我们看到预 token `' cat'` 在应用 BPE 合并后表示为 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`。最后一个预 token `' ate'` 在应用 BPE 合并后是 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此，对输入字符串编码的最终结果是 `[9, 7, 1, 5, 10, 3]`。

**特殊 token**

你的分词器在编码文本时应该能够正确处理用户定义的特殊 token（在构造分词器时提供）。

**内存考虑**

假设我们要对一个大到无法放入内存的文本文件进行分词。为了高效地对大文件（或任何其他数据流）进行分词，我们需要将其分解为可管理的块，并依次处理每个块，使得内存复杂度是常数级别，而不是随文本大小线性增长。在这样做时，我们需要确保 token 不跨块边界，否则我们会得到与在内存中直接对整个序列分词的朴素方法不同的分词结果。

#### 2.6.2 解码文本

要将整数 token ID 序列解码回原始文本，我们只需在词汇表中查找每个 ID 对应的条目（一个字节序列），将它们拼接在一起，然后将字节解码为 Unicode 字符串。请注意，输入的 ID 并不保证能映射为有效的 Unicode 字符串（因为用户可以输入任意整数 ID 序列）。如果输入的 token ID 无法产生有效的 Unicode 字符串，你应该用官方的 Unicode 替换字符 U+FFFD 替换格式错误的字节。`bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 会自动用替换标记替换格式错误的数据。

**问题 (tokenizer)：实现分词器（15 分）**

**交付物：** 实现一个 `Tokenizer` 类，给定词汇表和合并列表，将文本编码为整数 ID，并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token（如果它们还不在词汇表中，则将它们追加到词汇表中）。我们推荐以下接口：

```python
def __init__(self, vocab, merges, special_tokens=None)
```
从给定的词汇表、合并列表和（可选的）特殊 token 列表构造分词器。此函数应接受以下参数：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```
类方法，从序列化的词汇表和合并列表（与你的 BPE 训练代码输出的格式相同）以及（可选的）特殊 token 列表构造并返回一个 `Tokenizer`。此方法应接受以下额外参数：

- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```
将输入文本编码为 token ID 序列。

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```
给定一个字符串可迭代对象（例如 Python 文件句柄），返回一个惰性产生 token ID 的生成器。这对于我们无法直接加载到内存的大文件的内存高效分词是必需的。

```python
def decode(self, ids: list[int]) -> str
```
将 token ID 序列解码为文本。

要针对我们提供的测试测试你的 `Tokenizer`，你首先需要实现 [adapters.get_tokenizer] 处的测试适配器。然后，运行 `uv run pytest tests/test_tokenizer.py`。你的实现应该能够通过所有测试。

### 2.7 实验

**问题 (tokenizer_experiments)：分词器实验（4 分）**

(a) 从 TinyStories 和 OpenWebText 中各采样 10 个文档。使用你之前训练的 TinyStories 和 OpenWebText 分词器（词汇表大小分别为 10K 和 32K），将这些采样文档编码为整数 ID。每个分词器的压缩率（字节/token）是多少？

交付物：一到两句话的回答。

(b) 如果你用 TinyStories 分词器对 OpenWebText 样本进行分词会发生什么？比较压缩率和/或定性描述发生了什么。

交付物：一到两句话的回答。

(c) 估计你的分词器的吞吐量（例如，以字节/秒为单位）。对 Pile 数据集（825GB 文本）进行分词需要多长时间？

交付物：一到两句话的回答。

(d) 使用你的 TinyStories 和 OpenWebText 分词器，将相应的训练集和开发集编码为整数 token ID 序列。我们稍后会用它来训练我们的语言模型。我们建议将 token ID 序列化为数据类型为 `uint16` 的 NumPy 数组。为什么 `uint16` 是一个合适的选择？

交付物：一到两句话的回答。

---

## 3 Transformer 语言模型架构

语言模型将批量化的整数 token ID 序列作为输入（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），并返回词汇表上的（批量化）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch 张量），其中对于每个输入 token，预测的分布是关于下一个词的分布。在训练语言模型时，我们使用这些下一词预测来计算实际下一词与预测下一词之间的交叉熵损失。在推理期间从语言模型生成文本时，我们取最后一个时间步的预测下一词分布（即序列中的最后一项）来生成序列中的下一个 token（例如，通过取概率最高的 token、从分布中采样等），将生成的 token 添加到输入序列中，然后重复此过程。

在作业的这一部分中，你将从头开始构建这个 Transformer 语言模型。我们将从模型的高级描述开始，然后逐步详细介绍各个组件。

### 3.1 Transformer LM

给定一个 token ID 序列，Transformer 语言模型使用输入嵌入将 token ID 转换为稠密向量，将嵌入后的 token 通过 `num_layers` 个 Transformer 块，然后应用一个学习到的线性投影（"输出嵌入"或"LM 头"）来产生预测的下一 token logits。示意图见图 1。

```
Transformer Block
Token          ┌──────────────────┐
Embedding  →   │ Transformer Block │  × num_layers
Inputs         └──────────────────┘
                      ↓
                   ┌──────┐    ┌──────────┐    ┌─────────┐
                   │ Norm │ →  │  Linear   │ →  │ Softmax │ → Output Probabilities
                   └──────┘    │(Output    │    └─────────┘
                                │Embedding) │
                                └──────────┘
```

*图 1：我们的 Transformer 语言模型概览。*

```
输入张量，形状为 (batch_size, seq_len, d_model)
        │
   ┌────┴────┐
   │  Norm   │
   └────┬────┘
        │
  ┌─────┴──────────┐
  │ Causal Multi-  │
  │ Head Self-     │
  │ Attention      │
  │ w/ RoPE        │
  └─────┬──────────┘
        │
   ┌────┴────┐
   │   Add   │ ←── (残差连接)
   └────┬────┘
        │
   ┌────┴────┐
   │  Norm   │
   └────┬────┘
        │
  ┌─────┴──────────┐
  │ Position-Wise  │
  │ Feed-Forward   │
  └─────┬──────────┘
        │
   ┌────┴────┐
   │   Add   │ ←── (残差连接)
   └────┬────┘
        ↓
输出张量，形状为 (batch_size, seq_len, d_model)
```

*图 2：pre-norm Transformer 块。*

#### 3.1.0.1 Token 嵌入

在第一步中，Transformer 将（批量化的）token ID 序列嵌入到包含 token 身份信息的向量序列中（图 1 中的红色块）。

更具体地说，给定一个 token ID 序列，Transformer 语言模型使用 token 嵌入层来产生向量序列。每个嵌入层接收一个形状为 `(batch_size, sequence_length)` 的整数张量，并产生形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

#### 3.1.0.2 Pre-norm Transformer 块

嵌入之后，激活值由几个结构相同的神经网络层处理。标准的仅解码器 Transformer 语言模型由 `num_layers` 个相同的层组成（通常称为 Transformer"块"）。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块通过自注意力跨序列聚合信息，并（通过前馈层）对其进行非线性变换。

在 `num_layers` 个 Transformer 块之后，我们将取最终的激活值并将其转换为词汇表上的分布。

我们将实现"pre-norm" Transformer 块（详见 3.4 节），这需要在最终的 Transformer 块之后额外使用层归一化（详见下文），以确保其输出被正确缩放。

在此归一化之后，我们将使用标准的学习到的线性变换，将 Transformer 块的输出转换为预测的下一 token logits（参见例如 A. Radford et al. [7] 公式 2）。

### 3.2 备注：批处理、Einsum 和高效计算

在整个 Transformer 中，我们将对许多类似批量的输入执行相同的计算。以下是一些例子：

- **批次的元素：** 我们对每个批次元素应用相同的 Transformer 前向操作。
- **序列长度：** 诸如 RMSNorm 和前馈之类的"逐位置"操作对序列的每个位置都进行相同的操作。
- **注意力头：** 注意力操作在"多头"注意力操作中跨注意力头批量化。

有一种符合人体工程学的方式来执行这些操作是很有用的，这种方式既能充分利用 GPU，又易于阅读和理解。许多 PyTorch 操作可以在张量开头接收额外的"类批量"维度，并在这些维度上高效地重复/广播操作。

例如，假设我们正在做一个逐位置的批量操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" $D$，并且我们希望对形状为 `(d_model, d_model)` 的矩阵 $A$ 进行批量向量-矩阵乘法。在这种情况下，`D @ A` 将执行批量矩阵乘法，这是 PyTorch 中的一个高效原语，其中 `(batch_size, sequence_length)` 维度被批量化。

因此，假定你的函数可能被给予额外的类批量维度，并将这些维度保留在 PyTorch 形状的开头是有帮助的。为了组织张量以便能够以这种方式批量化，它们可能需要通过许多 `view`、`reshape` 和 `transpose` 步骤来整形。这可能有点麻烦，而且通常很难阅读代码在做什么以及张量的形状是什么。

一个更符合人体工程学的选择是在 `torch.einsum` 中使用 einsum 表示法，或者使用像 `einops` 或 `einx` 这样的框架无关库。两个关键操作是 `einsum`（可以对输入张量的任意维度进行张量收缩）和 `rearrange`（可以对任意维度进行重排、拼接和拆分）。事实证明，机器学习中几乎所有的操作都是维度杂耍和张量收缩的某种组合，外加偶尔的（通常是逐点的）非线性函数。这意味着使用 einsum 表示法时，你的很多代码可以更具可读性和灵活性。

我们强烈建议你在课堂上学习和使用 einsum 表示法。以前没有接触过 einsum 表示法的学生应该使用 einops（文档在此），已经熟悉 einops 的学生应该学习更通用的 einx（在此）。这两个包都已经安装在我们提供的环境中。

下面我们给出一些如何使用 einsum 表示法的例子。这些是对 einops 文档的补充，你应该先阅读文档。

**示例 (einstein_example1)：使用 einops.einsum 的批量矩阵乘法**

```python
import torch
from einops import rearrange, einsum

## 基本实现
Y = D @ A.T
# 很难看出输入和输出的形状以及它们的含义。
# D 和 A 可以有什么形状，其中是否有任何意外行为？

## Einsum 是自文档化且健壮的
#                          D                A     ->          Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## 或者，一个批量版本，其中 D 可以有任意前导维度，但 A 受约束。
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**示例 (einstein_example2)：使用 einops.rearrange 的广播操作**

我们有一批图像，对于每个图像，我们想根据某个缩放因子生成 10 个变暗的版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## 整形并相乘
dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## 或者一步完成：
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

**示例 (einstein_example3)：使用 einops.rearrange 的像素混合**

假设我们有一批表示为形状 `(batch, height, width, channel)` 的张量的图像，并且我们想对图像的所有像素执行线性变换，但此变换应该对每个通道独立进行。我们的线性变换表示为形状为 `(height * width, height * width)` 的矩阵 $B$。

```python
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## 重排图像张量以进行跨所有像素的混合
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

使用 einops 代替：

```python
height = width = 32
## rearrange 替代了笨重的 torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

或者，如果你感觉疯狂的话：使用 `einx.dot`（einx 相当于 einops.einsum）一步完成：

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

这里的第一个实现可以通过在前后添加注释来表明输入和输出的形状，但这很笨拙且容易出错。使用 einsum 表示法，文档就是实现！

Einsum 表示法可以处理任意的输入批量化维度，但还有一个关键好处是自文档化。在使用 einsum 表示法的代码中，输入和输出张量的相关形状要清楚得多。对于其余的张量，你可以考虑使用 Tensor 类型提示，例如使用 jaxtyping 库（不特定于 JAX）。

我们将在作业 2 中更多地讨论使用 einsum 表示法的性能影响，但现在要知道它们几乎总是比替代方案更好！

#### 3.2.1 数学符号和内存排序

许多机器学习论文在其符号中使用行向量，这产生了与 NumPy 和 PyTorch 默认使用的行主内存排序良好配合的表示。使用行向量，线性变换看起来像这样：

$$y = xW^\top, \tag{1}$$

其中行主序 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，行向量 $x \in \mathbb{R}^{1 \times d_{in}}$。请注意，这让我们可以通过增加 $x$ 的最外层维度来批量化输入，这意味着我们可以用矩阵输入 $X \in \mathbb{R}^{batch \times d_{in}}$ 替换向量输入 $x$。

在线性代数中，通常更常用列向量，其中线性变换看起来像这样：

$$y = Wx, \tag{2}$$

给定行主序 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 和列向量 $x \in \mathbb{R}^{d_{in}}$。在这种情况下要对输入进行批量化，$x$ 的批次维度必须放在最后，因此 $x$ 需要替换为矩阵 $\tilde{X} \in \mathbb{R}^{d_{in} \times batch}$。

在本作业中，我们将主要使用列向量作为数学符号，因为数学通常遵循这种符号。你应该记住，如果你想使用普通的矩阵乘法符号，你将必须按照行向量约定在公式 1 中看到的那样对矩阵应用转置，因为 PyTorch 使用行主内存排序。如果你使用 einsum 进行线性代数运算，只要正确标记轴，这应该不是问题。顺便说一句，值得注意的是，其他语言/线性代数包（如 Matlab、Julia 和 Fortran）都使用列主内存排序，这意味着批量化维度放在最后，但 Python 及相关包采用了 C 标准的行主序。

### 3.3 基本构建块：Linear 和 Embedding 模块

#### 3.3.1 参数初始化

有效地训练神经网络通常需要仔细初始化模型参数——糟糕的初始化会导致不良行为，例如梯度消失或爆炸。Pre-norm Transformer 对初始化异常鲁棒，但它们仍然可以对训练速度和收敛产生重大影响。由于本作业已经很长，我们将把细节留到作业 3，而是给你一些应该在大多数情况下都能很好工作的近似初始化。现在，使用：

- **Linear 权重：** $\mathcal{N}(\mu = 0, \sigma^2 = \frac{2}{d_{in} + d_{out}})$，截断于 $[-3\sigma, 3\sigma]$。
- **Embedding：** $\mathcal{N}(\mu = 0, \sigma^2 = 1)$，截断于 $[-3, 3]$
- **RMSNorm：** $\mathbb{1}$

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.3.2 Linear 模块

Linear 层是 Transformer 和一般神经网络的基本构建块。首先，你将实现自己的 `Linear` 类，它继承自 `torch.nn.Module` 并执行线性变换：

$$y = Wx. \tag{3}$$

请注意，我们不包含偏置项，这与大多数现代 LLM 一致。

**问题 (linear)：实现 Linear 模块（1 分）**

**交付物：** 实现一个继承自 `torch.nn.Module` 并执行线性变换的 `Linear` 类。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，除了没有偏置参数。我们推荐以下接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```
构造一个线性变换模块。此函数应接受以下参数：

- `in_features: int` 输入的最终维度
- `out_features: int` 输出的最终维度
- `device: torch.device | None = None` 存储参数的设备
- `dtype: torch.dtype | None = None` 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```
对输入应用线性变换。

确保：

- 继承 `nn.Module`
- 调用超类构造函数
- 将你的参数构造并存储为 $W$（而不是 $W^\top$），将其放在 `nn.Parameter` 中
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`

对于初始化，使用上述设置以及 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试你的 Linear 模块，请实现 [adapters.run_linear] 处的测试适配器。适配器应将给定的权重加载到你的 Linear 模块中。你可以使用 `Module.load_state_dict` 来实现。然后，运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

如上所述，Transformer 的第一层是一个嵌入层，它将整数 token ID 映射到维度为 `d_model` 的向量空间。我们将实现一个继承自 `torch.nn.Module` 的自定义 `Embedding` 类（因此你不应使用 `nn.Embedding`）。`forward` 方法应通过使用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token ID 对形状为 `(vocab_size, d_model)` 的嵌入矩阵进行索引来选择每个 token ID 的嵌入向量。

**问题 (embedding)：实现 Embedding 模块（1 分）**

**交付物：** 实现继承自 `torch.nn.Module` 并执行嵌入查找的 `Embedding` 类。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```
构造一个嵌入模块。此函数应接受以下参数：

- `num_embeddings: int` 词汇表的大小
- `embedding_dim: int` 嵌入向量的维度，即 $d_{model}$
- `device: torch.device | None = None` 存储参数的设备
- `dtype: torch.dtype | None = None` 参数的数据类型

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```
查找给定 token ID 的嵌入向量。

确保：

- 继承 `nn.Module`
- 调用超类构造函数
- 将你的嵌入矩阵初始化为 `nn.Parameter`
- 存储嵌入矩阵时使 `d_model` 为最终维度
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`

再次，使用上述设置进行初始化，并使用 `torch.nn.init.trunc_normal_` 初始化权重。

要测试你的实现，请实现 [adapters.run_embedding] 处的测试适配器。然后，运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer 块

每个 Transformer 块有两个子层：一个多头自注意力机制和一个逐位置前馈网络（[A. Vaswani et al., 2017]，第 3.1 节）。

在原始 Transformer 论文中，模型在两个子层中的每一个周围使用残差连接，然后进行层归一化。这种架构通常被称为"post-norm" Transformer，因为层归一化应用于子层的输出。然而，各种研究发现，将层归一化从每个子层的输出移动到每个子层的输入（并在最终的 Transformer 块之后添加额外的层归一化）可以提高 Transformer 训练的稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]——见图 2 对这种"pre-norm" Transformer 块的可视化表示。然后，每个 Transformer 块子层的输出通过残差连接加到子层输入上（A. Vaswani et al. [8]，第 5.4 节）。pre-norm 的一个直觉是，存在一条从输入嵌入到 Transformer 最终输出的干净"残差流"，中间没有任何归一化，这被认为可以改善梯度流动。这种 pre-norm Transformer 现在是当今语言模型中使用的标准（例如 GPT-3、LLaMA、PaLM 等），因此我们将实现这种变体。我们将逐步介绍 pre-norm Transformer 块的每个组件，并依次实现它们。

#### 3.4.1 均方根层归一化（RMSNorm）

A. Vaswani et al. [8] 的原始 Transformer 实现使用层归一化 [J. L. Ba et al., 2016] 来归一化激活值。遵循 H. Touvron et al. [12]，我们将使用均方根层归一化（RMSNorm；B. Zhang et al. [13]，公式 4）进行层归一化。给定激活向量 $a \in \mathbb{R}^{d_{model}}$，RMSNorm 将按如下方式重新缩放每个激活 $a_i$：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i, \tag{4}$$

其中 $\text{RMS}(a) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}$。这里，$g_i$ 是可学习的"增益"参数（总共有 $d_{model}$ 个这样的参数），$\varepsilon$ 是一个超参数，通常固定为 1e-5。

你应该将输入向上转换为 `torch.float32`，以防止对输入求平方时溢出。总的来说，你的 `forward` 方法应如下所示：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 你的代码在这里执行 RMSNorm
...
result = ...
# 以原始 dtype 返回结果
return result.to(in_dtype)
```

**问题 (rmsnorm)：均方根层归一化（1 分）**

**交付物：** 将 RMSNorm 实现为 `torch.nn.Module`。我们推荐以下接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```
构造 RMSNorm 模块。此函数应接受以下参数：

- `d_model: int` 模型的隐藏维度
- `eps: float = 1e-5` 用于数值稳定性的 epsilon 值
- `device: torch.device | None = None` 存储参数的设备
- `dtype: torch.dtype | None = None` 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```
处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，并返回相同形状的张量。

注意：请记住在执行归一化之前将输入向上转换为 `torch.float32`（然后向下转换回原始 dtype），如上所述。

要测试你的实现，请实现 [adapters.run_rmsnorm] 处的测试适配器。然后，运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

```
 4 │                    │  SiLU(x)
 2 │      SiLU          │  Identity: f(x) = x
 0 │──────────          │  ReLU: f(x) = max(0, x)
-2 │                    │
-4 │                    │
   └──────────          │
        x
```

*图 3：SiLU（又称 Swish）和 ReLU 激活函数的比较。*

在原始 Transformer 论文（A. Vaswani et al. [8] 第 3.3 节）中，Transformer 前馈网络由两个线性变换组成，它们之间有一个 ReLU 激活函数（$\text{ReLU}(x) = \max(0, x)$）。在那种原始架构中，内部前馈层的维度通常是输入维度的 4 倍。

然而，与这种原始设计相比，现代语言模型往往包含两个主要变化：它们使用另一种激活函数并采用门控机制。具体来说，我们将实现 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 采用的"SwiGLU"激活函数，它将 SiLU（通常称为 Swish）激活与称为门控线性单元（GLU）的门控机制相结合。我们还将省略线性层中有时使用的偏置项，遵循自 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来的大多数现代 LLM。

SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} \tag{5}$$

如图 3 所示，SiLU 激活函数类似于 ReLU 激活函数，但在零处是平滑的。

门控线性单元（GLU）最初由 Y. N. Dauphin et al. [19] 定义为通过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x, \tag{6}$$

其中 $\odot$ 表示逐元素乘法。门控线性单元被认为"通过为梯度提供线性路径，同时保留非线性能力，减少了深度架构的梯度消失问题"。

将 SiLU/Swish 和 GLU 放在一起，我们得到 SwiGLU，我们将用它来构建前馈网络：

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1 x) \odot W_3 x), \tag{7}$$

其中 $x \in \mathbb{R}^{d_{model}}$，$W_1, W_3 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$，并且通常 $d_{ff} = \frac{8}{3} d_{model}$。对于具体实现，将其四舍五入到附近的 64 的倍数是可以的，以提高硬件效率。

N. Shazeer [20] 首先提出将 SiLU/Swish 激活与 GLU 相结合，并进行了实验，表明 SwiGLU 在语言建模任务上优于 ReLU 和 SiLU（无门控）等基线。在作业后面，你将比较 SwiGLU 和 SiLU。虽然我们提到了这些组件的一些启发式论证（论文提供了更多支持证据），但最好保持实证的视角：Shazeer 论文中一句现在著名的话是：

> "我们不解释为什么这些架构似乎有效；我们将它们的成功归因于，如同其他一切，神圣的仁慈。"

**问题 (positionwise_feedforward)：实现逐位置前馈网络（2 分）**

**交付物：** 实现 SwiGLU 前馈网络，由 SiLU 激活函数和 GLU 组成。

注意：在这种特殊情况下，你可以在你的实现中随意使用 `torch.sigmoid` 以获得数值稳定性。

你应该在你的实现中将 $d_{ff}$ 设置为大约 $\frac{8}{3} \times d_{model}$，同时确保内部前馈层的维度是 64 的倍数，以充分利用你的硬件。要针对我们提供的测试测试你的实现，你需要实现 [adapters.run_swiglu] 处的测试适配器。然后，运行 `uv run pytest -k test_swiglu` 来测试你的实现。

#### 3.4.3 相对位置嵌入

为了将位置信息注入模型，我们将实现旋转位置嵌入 [J. Su et al., 2021]，通常称为 RoPE。对于 token 位置 $i$ 处的给定查询 token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们将应用成对旋转矩阵 $R_i$，得到 $q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}$。这里，$R_i$ 将把嵌入元素对 $q^{(i)}_{2k-1:2k}$ 作为 2D 向量旋转角度 $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$，其中 $k \in \{1, \ldots, d/2\}$，$\Theta$ 是某个常数。因此，我们可以将 $R_i$ 视为大小为 $d \times d$ 的块对角矩阵，块为 $R_i^k$，其中 $k \in \{1, \ldots, \frac{d}{2}\}$，且：

$$R_i^k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix} \tag{8}$$

因此我们得到完整的旋转矩阵：

$$R_i = \begin{pmatrix} R_i^1 & 0 & 0 & \cdots & 0 \\ 0 & R_i^2 & 0 & \cdots & 0 \\ 0 & 0 & R_i^3 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R_i^{d/2} \end{pmatrix}, \tag{9}$$

其中 0 表示 $2 \times 2$ 零矩阵。虽然可以构造完整的 $d \times d$ 矩阵，但好的解决方案应该利用这个矩阵的性质来更高效地实现变换。由于我们只关心给定序列内 token 的相对旋转，我们可以跨层和不同批次重用我们为 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 计算的值。如果你想优化它，你可以使用一个被所有层引用的单个 RoPE 模块，它可以在 `__init__` 期间使用 `self.register_buffer(persistent=False)` 创建一个 2D 预计算的 sin 和 cos 值缓冲区，而不是 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。然后对 $k^{(j)}$ 执行我们对 $q^{(i)}$ 所做的完全相同的旋转过程，按相应的 $R_j$ 旋转。请注意，这一层没有可学习的参数。

**问题 (rope)：实现 RoPE（2 分）**

**交付物：** 实现一个将 RoPE 应用于输入张量的 `RotaryPositionalEmbedding` 类。

推荐以下接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```
构造 RoPE 模块并在需要时创建缓冲区。

- `theta: float` RoPE 的 $\Theta$ 值
- `d_k: int` 查询和键向量的维度
- `max_seq_len: int` 将输入的最大序列长度
- `device: torch.device | None = None` 存储缓冲区的设备

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```
处理形状为 `(..., seq_len, d_k)` 的输入张量并返回相同形状的张量。请注意，你应该容忍具有任意数量批次维度的 $x$。你应该假设 token 位置是一个形状为 `(..., seq_len)` 的张量，指定 $x$ 沿序列维度的 token 位置。

你应该使用 token 位置沿序列维度切片你的（可能是预计算的）cos 和 sin 张量。

要测试你的实现，请完成 [adapters.run_rope] 并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

我们现在将实现 A. Vaswani et al. [8]（第 3.2.1 节）中描述的缩放点积注意力。作为预备步骤，Attention 操作的定义将使用 softmax，这是一种将未归一化的分数向量转换为归一化分布的操作：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}. \tag{10}$$

请注意，对于大值，$\exp(v_i)$ 可能变为 inf（然后，$\frac{\text{inf}}{\text{inf}} = \text{NaN}$）。我们可以通过注意到 softmax 操作对向所有输入添加任何常数 $c$ 是不变的来避免这种情况。我们可以利用这个特性来获得数值稳定性——通常，我们将从 $v$ 的所有元素中减去 $v$ 的最大条目，使新的最大条目为 0。现在你将使用这个技巧来实现 softmax，以获得数值稳定性。

**问题 (softmax)：实现 softmax（1 分）**

**交付物：** 编写一个函数，对张量应用 softmax 操作。你的函数应该接受两个参数：一个张量和一个维度 $i$，并将 softmax 应用于输入张量的第 $i$ 个维度。输出张量应与输入张量具有相同的形状，但其第 $i$ 个维度现在将具有归一化的概率分布。使用从第 $i$ 个维度的所有元素中减去第 $i$ 个维度中的最大值的技巧，以避免数值稳定性问题。

要测试你的实现，请完成 [adapters.run_softmax] 并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在我们可以将 Attention 操作数学定义如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \tag{11}$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里，$Q$、$K$ 和 $V$ 都是此操作的输入——请注意，这些不是可学习的参数。

**掩码：** 有时掩码注意力操作的输出很方便。掩码应具有形状 $M \in \{\text{True}, \text{False}\}^{n \times m}$，此布尔矩阵的每一行 $i$ 指示查询 $i$ 应关注哪些键。通常（而且有点令人困惑），位置 $(i, j)$ 处的值为 True 表示查询 $i$ 确实关注键 $j$，值为 False 表示查询不关注键。换句话说，"信息流动"在值为 True 的 $(i, j)$ 对处。例如，考虑一个 $1 \times 3$ 的掩码矩阵，条目为 `[[True, True, False]]`。单个查询向量只关注前两个键。

在计算上，使用掩码将比对子序列计算注意力高效得多，我们可以通过取 pre-softmax 值（$QK^\top / \sqrt{d_k}$）并向掩码矩阵中为 False 的任何条目添加 $-\infty$ 来做到这一点。

**问题 (scaled_dot_product_attention)：实现缩放点积注意力（5 分）**

**交付物：** 实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任意数量的其他类批次维度（如果提供）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。有关类批次维度的讨论，请参见第 3.2 节。

你的实现还应支持形状为 `(seq_len, seq_len)` 的可选用户提供的布尔掩码。掩码值为 True 的位置的注意力概率应总计为 1，掩码值为 False 的位置的注意力概率应为零。

要针对我们提供的测试测试你的实现，你需要实现 [adapters.run_scaled_dot_product_attention] 处的测试适配器。`uv run pytest -k test_scaled_dot_product_attention` 在三阶输入张量上测试你的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 在四阶输入张量上测试你的实现。

#### 3.4.5 因果多头自注意力

我们将实现 A. Vaswani et al. [8] 第 3.2.2 节中描述的多头自注意力。回想一下，从数学上讲，应用多头注意力的操作定义如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \tag{12}$$

其中：

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) \tag{13}$$

$Q_i$、$K_i$、$V_i$ 分别是 $Q$、$K$ 和 $V$ 沿嵌入维度大小为 $d_k$ 或 $d_v$ 的第 $i \in \{1, \ldots, h\}$ 个切片。Attention 是第 3.4.4 节中定义的缩放点积注意力操作。由此我们可以形成多头自注意力操作：

$$\text{MultiHeadSelfAttention}(x) = W_O \text{MultiHead}(W_Q x, W_K x, W_V x) \tag{14}$$

这里，可学习的参数是 $W_Q \in \mathbb{R}^{hd_k \times d_{model}}$，$W_K \in \mathbb{R}^{hd_k \times d_{model}}$，$W_V \in \mathbb{R}^{hd_v \times d_{model}}$，以及 $W_O \in \mathbb{R}^{d_{model} \times hd_v}$。由于 $Q$、$K$ 和 $V$ 在多头注意力操作中被切片，我们可以将 $W_Q$、$W_K$ 和 $W_V$ 视为沿输出维度对每个头分开。当你完成这个工作时，你应该在总共三次矩阵乘法中计算键、值和查询投影。⁵

⁵作为延伸目标，尝试将键、查询和值投影组合成单个权重矩阵，这样你只需要一次矩阵乘法。

**因果掩码**

你的实现应防止模型关注序列中的未来 token。换句话说，如果模型被给定一个 token 序列 $t_1, \ldots, t_n$，并且我们想要计算前缀 $t_1, \ldots, t_i$（其中 $i < n$）的下一词预测，则模型不应能够访问（关注）位置 $t_{i+1}, \ldots, t_n$ 处的 token 表示，因为在推理期间生成文本时它将无法访问这些 token（而且这些未来 token 会泄露关于真实下一词身份的信息，从而使语言建模预训练目标变得平凡）。对于输入 token 序列 $t_1, \ldots, t_n$，我们可以通过运行多头自注意力 $n$ 次（对于序列中的 $n$ 个唯一前缀）来简单地阻止对未来 token 的访问。相反，我们将使用因果注意力掩码，它允许 token $i$ 关注序列中的所有位置 $j \leq i$。你可以使用 `torch.triu` 或广播索引比较来构造这个掩码，并且你应该利用你在第 3.4.4 节中的缩放点积注意力实现已经支持注意力掩码这一事实。

**应用 RoPE**

RoPE 应应用于查询和键向量，但不应用于值向量。此外，头维度应作为批次维度处理，因为在多头注意力中，注意力是对每个头独立应用的。这意味着对每个头的查询和键向量应用完全相同的 RoPE 旋转。

**问题 (multihead_self_attention)：实现因果多头自注意力（5 分）**

**交付物：** 将因果多头自注意力实现为 `torch.nn.Module`。你的实现应接受（至少）以下参数：

- `d_model: int` Transformer 块输入的维度。
- `num_heads: int` 多头自注意力中使用的头数。

遵循 A. Vaswani et al. [8]，设置 $d_k = d_v = \frac{d_{model}}{h}$。要针对我们提供的测试测试你的实现，请实现 [adapters.run_multihead_self_attention] 处的测试适配器。然后，运行 `uv run pytest -k test_multihead_self_attention` 来测试你的实现。

### 3.5 完整的 Transformer LM

让我们开始组装 Transformer 块（回顾图 2 会有所帮助）。一个 Transformer 块包含两个"子层"，一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。在每个子层中，我们首先执行 RMSNorm，然后是主要操作（MHA/FF），最后添加残差连接。

具体来说，Transformer 块的前半部分（第一个"子层"）应实现以下一组更新，以从输入 $x$ 产生输出 $y$：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)). \tag{15}$$

**问题 (transformer_block)：实现 Transformer 块（3 分）**

实现第 3.4 节所述并如图 2 所示的 pre-norm Transformer 块。你的 Transformer 块应接受（至少）以下参数：

- `d_model: int` Transformer 块输入的维度。
- `num_heads: int` 多头自注意力中使用的头数。
- `d_ff: int` 逐位置前馈内层的维度。

要测试你的实现，请实现适配器 [adapters.run_transformer_block]。然后运行 `uv run pytest -k test_transformer_block` 来测试你的实现。

**交付物：** 通过所提供测试的 Transformer 块代码。

现在我们将各块组装在一起，遵循图 1 的高级示意图。遵循我们在第 3.1.0.1 节中对嵌入的描述，将其馈入 `num_layers` 个 Transformer 块，然后将其传递到最终层归一化和 LM 头，以获得词汇表上的未归一化分布（logits）。

**问题 (transformer_lm)：实现 Transformer LM（3 分）**

是时候把所有东西放在一起了！如第 3.1 节所述并如图 1 所示实现 Transformer 语言模型。至少，你的实现应接受 Transformer 块的所有上述构造参数，以及这些额外参数：

- `vocab_size: int` 词汇表的大小，用于确定 token 嵌入矩阵的维度。
- `context_length: int` 最大上下文长度，用于确定 RoPE sin 和 cos 缓冲区的维度。
- `num_layers: int` 要使用的 Transformer 块数量。

要针对我们提供的测试测试你的实现，你首先需要实现 [adapters.run_transformer_lm] 处的测试适配器。然后，运行 `uv run pytest -k test_transformer_lm` 来测试你的实现。

**交付物：** 通过上述测试的 Transformer LM 模块。

#### 3.5.0.1 资源核算

理解 Transformer 的各个部分如何消耗计算和内存是很有用的。我们将逐步进行一些基本的"FLOPs 核算"。Transformer 中绝大多数 FLOPS 是矩阵乘法，因此我们的核心方法很简单：

1. 写下 Transformer 前向传播中的所有矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实将很有用：

**规则：** 给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，矩阵-矩阵乘积 $AB$ 需要 $2mnp$ 次 FLOPs。

要看到这一点，请注意 $(AB)[i, j] = A[i, :] \cdot B[:, j]$，并且这个点积需要 $n$ 次加法和 $n$ 次乘法（$2n$ 次 FLOPs）。然后，由于矩阵-矩阵乘积 $AB$ 有 $m \times p$ 个条目，FLOPS 总数为 $(2n)(mp) = 2mnp$。

现在，在你做下一个问题之前，浏览 Transformer 块和 Transformer LM 的每个组件，并列出所有矩阵乘法及其相关的 FLOPs 成本会很有帮助。

**问题 (transformer_accounting)：Transformer LM 资源核算（5 分）**

(a) 考虑使用我们作业架构的 GPT-2 XL 大小的模型，其配置如下：

- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 4,288（最接近 $\frac{8}{3} \times 1,600$ 的 64 的倍数）

假设我们使用此配置构建了模型。我们的模型有多少可训练参数？假设每个参数使用单精度浮点表示，仅加载此模型需要多少内存？

**交付物：** 一到两句话的回答。

(b) 确定完成我们 GPT-2 XL 形状模型前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列有 `context_length` 个 token。

**交付物：** 矩阵乘法列表（带描述）以及所需的 FLOPs 总数。

(c) 根据你上面的分析，模型的哪些部分需要最多的 FLOPs？

**交付物：** 一到两句话的回答。

(d) 用 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复你的分析。随着模型大小的增加，Transformer LM 的哪些部分在总 FLOPs 中所占比例会增加或减少？

**交付物：** 对于每个模型，提供模型组件的细分及其相关的 FLOPs（作为前向传播所需总 FLOPs 的比例）。此外，提供一到两句话的描述，说明改变模型大小如何改变每个组件的比例 FLOPs。

(e) 取 GPT-2 XL 并将上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？模型组件的 FLOPs 相对贡献如何变化？

**交付物：** 一到两句话的回答。

---

## 4 训练一个 Transformer LM

我们现在有了预处理数据的步骤（通过分词器）和模型（Transformer）。剩下的就是构建所有支持训练的代码。这包括：
- **损失：** 我们需要定义损失函数（交叉熵）。
- **优化器：** 我们需要定义优化器以最小化此损失（AdamW）。
- **训练循环：** 我们需要所有支持基础设施，用于加载数据、保存检查点和管理训练。

### 4.1 交叉熵损失
回想一下，Transformer 语言模型为每个长度为 𝑚 + 1 的序列 𝑥 和 𝑖 = 1, …, 𝑚 定义一个分布 𝑝_𝜃(𝑥_{𝑖+1} | 𝑥_{1:𝑖})。给定一个由长度为 𝑚 + 1 的序列组成的训练集 𝐷，我们定义标准的交叉熵（负对数似然）损失函数：
$$\ell(𝜃; 𝐷) = \frac{1}{|𝐷| m} \sum_{𝑥∈𝐷} \sum_{i=1}^m −\log 𝑝_𝜃(𝑥_{𝑖+1} | 𝑥_{1:𝑖})。 \quad (16)$$
（注意，Transformer 中的一次前向传播为所有 𝑖 = 1, …, 𝑚 产生 𝑝_𝜃(𝑥_{𝑖+1} | 𝑥_{1:𝑖})。）

特别是，Transformer 为每个位置 𝑖 计算 logits 𝑜_𝑖 ∈ ℝ^{vocab_size}，这导致：^6
$$𝑝(𝑥_{𝑖+1} | 𝑥_{1:𝑖}) = softmax(𝑜_𝑖)[𝑥_{𝑖+1}] = \frac{\exp(𝑜_𝑖[𝑥_{𝑖+1}])}{\sum_{a=1}^{vocab\_size} \exp(𝑜_𝑖[𝑎])}。 \quad (17)$$

交叉熵损失通常相对于 logits 向量 𝑜_𝑖 ∈ ℝ^{vocab_size} 和目标 𝑥_{𝑖+1} 定义。^7

^6 注意，𝑜_𝑖[𝑘] 指向量 𝑜_𝑖 在索引 𝑘 处的值。
^7 这对应于 𝑥_{𝑖+1} 上的 Dirac delta 分布与预测的 softmax(𝑜_𝑖) 分布之间的交叉熵。

实现交叉熵损失需要对数值问题小心处理，就像 softmax 的情况一样。

**问题（cross_entropy）：实现交叉熵（1 分）**
交付物：编写一个计算交叉熵损失的函数，它接受预测的 logits（𝑜_𝑖）和目标（𝑥_{𝑖+1}），并计算交叉熵 ℓ_𝑖 = −log softmax(𝑜_𝑖)[𝑥_{𝑖+1}]。你的函数应处理以下内容：
- 为数值稳定性减去最大元素。
- 尽可能抵消 log 和 exp。
- 处理任何额外的批处理维度，并返回跨批次的平均值。与第 3.2 节一样，我们假设类似批处理的维度总是放在最前面，在词汇表大小维度之前。

实现 [adapters.run_cross_entropy]，然后运行 `uv run pytest -k test_cross_entropy` 来测试你的实现。

**困惑度**
交叉熵足以用于训练，但当我们评估模型时，我们还想报告困惑度。对于长度为 𝑚 的序列，我们遭受交叉熵损失 ℓ_1, …, ℓ_𝑚：
$$perplexity = \exp(\frac{1}{m} \sum_{i=1}^m \ell_i)。 \quad (18)$$

### 4.2 SGD 优化器
现在我们有了损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 𝜃_0 开始。然后对于每一步 𝑡 = 0, …, 𝑇 − 1，我们执行以下更新：
$$𝜃_{𝑡+1} \leftarrow 𝜃_𝑡 − 𝛼_𝑡 ∇𝐿(𝜃_𝑡; 𝐵_𝑡)， \quad (19)$$
其中 𝐵_𝑡 是从数据集 𝐷 中采样的随机批次，学习率 𝛼_𝑡 和批次大小 |𝐵_𝑡| 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD
为了实现我们的优化器，我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。一个 Optimizer 子类必须实现两个方法：
- `def __init__(self, params, ...)` 应初始化你的优化器。这里，params 是要优化的参数集合（如果用户想为模型的不同部分使用不同的超参数（如学习率），也可以是参数组）。确保将 params 传给基类的 `__init__` 方法，基类会存储这些参数供 step 使用。你可以根据优化器接受额外的参数（例如，学习率是一个常见的参数），并将它们以字典形式传给基类构造函数，其中键是你为这些参数选择的名称（字符串）。
- `def step(self)` 应对参数做一次更新。在训练循环中，这将在反向传播之后被调用，所以你可以访问最后一批的梯度。此方法应遍历每个参数张量 p 并在原地修改它们，即设置 p.data，它保存与该参数关联的张量，基于梯度 p.grad（如果存在），即损失相对于该参数的梯度的张量。

PyTorch 优化器 API 有一些微妙之处，所以用一个例子来解释更容易。为了使例子更丰富，我们将实现 SGD 的一个轻微变体，其中学习率随训练衰减，从初始学习率 𝛼 开始，随时间采取越来越小的步长：
$$𝜃_{𝑡+1} = 𝜃_𝑡 − \sqrt{\frac{𝛼}{𝑡+1}} ∇𝐿(𝜃_𝑡; 𝐵_𝑡) \quad (20)$$

让我们看看这种 SGD 版本会如何实现为 PyTorch Optimizer：
```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]       # Get state associated with p.
                t = state.get("t", 0)       # Get iteration number from the state, or 0.
                grad = p.grad.data          # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad   # Update weight tensor in-place.
                state["t"] = t + 1          # Increment iteration number.
        return loss
```

在 `__init__` 中，我们将参数以及默认超参数传给基类构造函数（参数可能分组传入，每组有不同的超参数）。如果参数只是 `torch.nn.Parameter` 对象的单一集合，基构造函数将创建单个组并分配默认超参数。然后，在 step 中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用公式 20。这里，我们将迭代次数作为与每个参数关联的状态保存：我们首先读取此值，在梯度更新中使用它，然后更新它。

API 规定用户可能传入一个可调用的 closure 以在优化器步骤之前重新计算损失。我们不需要这个用于我们将使用的优化器，但我们添加它以符合 API。

为了看到这一点，我们可以使用以下训练循环的最小示例：
```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean()  # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward()  # Run backward pass, which computes gradients.
    opt.step()  # Run optimizer step.
```

这是训练循环的典型结构：在每次迭代中，我们将计算损失并运行优化器的一个步骤。当训练语言模型时，我们的可学习参数将来自模型（在 PyTorch 中，m.parameters() 给我们这个集合）。损失将在采样的数据批次上计算，但训练循环的基本结构是相同的。

**问题（learning_rate_tuning）：调优学习率（1 分）**
正如我们将看到的，对训练影响最大的超参数之一是学习率。让我们在我们的玩具示例中实际看看。使用三个其他学习率值运行上面的 SGD 示例：1e1、1e2 和 1e3，只运行 10 个训练迭代。这些学习率各自的损失会发生什么？它衰减得更快、更慢，还是发散（即随训练增加）？
交付物：一到两句话的回答，说明你观察到的行为。

### 4.3 AdamW
现代语言模型通常使用更复杂的优化器进行训练，而不是 SGD。最近使用的大多数优化器是 Adam 优化器 [D. P. Kingma 等人，2015] 的衍生。我们将使用 AdamW [I. Loshchilov 等人，2019]，它在最近的工作中被广泛使用。AdamW 提出了对 Adam 的修改，通过添加权重衰减（在每次迭代中，我们将参数向 0 拉近）以与梯度更新解耦的方式来改善正则化。我们将按照 I. Loshchilov 等人 [23] 的算法 2 实现 AdamW。

AdamW 是有状态的：对于每个参数，它保持其一阶和二阶矩的运行时估计。因此，AdamW 使用额外内存以换取改善的稳定性和收敛性。除了学习率 𝛼，AdamW 有一对超参数 (𝛽_1, 𝛽_2) 控制矩估计的更新，以及一个权重衰减率 𝜆。典型应用将 (𝛽_1, 𝛽_2) 设置为 (0.9, 0.999)，但像 LLaMA [H. Touvron 等人，2023] 和 GPT-3 [T. B. Brown 等人，2020] 这样的大型语言模型通常以 (0.9, 0.95) 训练。该算法可以写成如下形式，其中 𝜀 是一个小值（例如 10^−8），用于在我们可能在 𝑣 中得到极小值时提高数值稳定性：

**算法 1：AdamW 优化器**
1. init(𝜃) ▷ 初始化可学习参数
2. 𝑚 ← 0 ▷ 一阶矩向量的初始值；与 𝜃 形状相同
3. 𝑣 ← 0 ▷ 二阶矩向量的初始值；与 𝜃 形状相同
4. for 𝑡 = 1, …, 𝑇 do
5.    采样数据批次 𝐵_𝑡
6.    𝑔 ← ∇_𝜃 ℓ(𝜃; 𝐵_𝑡) ▷ 计算损失梯度
7.    𝛼_𝑡 ← 𝛼 \frac{\sqrt{1−𝛽_2^𝑡}}{1−𝛽_1^𝑡} ▷ 计算迭代 𝑡 的调整后 𝛼
8.    𝜃 ← 𝜃 − 𝛼𝜆𝜃 ▷ 应用权重衰减
9.    𝑚 ← 𝛽_1 𝑚 + (1 − 𝛽_1)𝑔 ▷ 更新一阶矩估计
10.   𝑣 ← 𝛽_2 𝑣 + (1 − 𝛽_2)𝑔² ▷ 更新二阶矩估计
11.   𝜃 ← 𝜃 − 𝛼_𝑡 \frac{𝑚}{\sqrt{𝑣}+𝜀} ▷ 应用矩调整的权重更新
12. end for

**算法 1：AdamW 优化器伪代码。** 注意 𝑡 从 1 开始。你现在将实现这个优化器。

**问题（adamw）：实现 AdamW（2 分）**
交付物：将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 𝛼，以及 𝛽、𝜀 和 𝜆 超参数。为了帮助你保持状态，基类 Optimizer 给你一个字典 self.state，它将 `nn.Parameter` 对象映射到一个字典，该字典存储你为该参数需要的任何信息（对于 AdamW，这将是矩估计）。实现 [adapters.get_adamw_cls] 并确保它通过 `uv run pytest -k test_adamw`。

**问题（adamw_accounting）：使用 AdamW 训练的资源核算（2 分）**
让我们计算运行 AdamW 需要多少内存和计算。假设我们对每个张量使用 float32。
(a) 运行 AdamW 需要多少峰值内存？根据参数、激活值、梯度和优化器状态的内存使用来分解你的答案。用批次大小和模型超参数（vocab_size、context_length、num_layers、d_model、num_heads）表示你的答案。假设 d_ff = 8/3 × d_model。
为简单起见，在计算激活值内存时，只考虑以下组件：
- Transformer 块
  - RMSNorm(s)
  - 多头自注意力子层：𝑄𝐾𝑉 投影、𝑄𝐾^𝑇 矩阵乘法、softmax、值的加权和、输出投影。
  - 位置级前馈（SwiGLU）：𝑊_1、𝑊_2、门控分支上的 SiLU、逐元素乘积、𝑊_3
- 最终 RMSNorm
- 输出嵌入
- logits 上的交叉熵
交付物：参数、激活值、梯度和优化器状态各自的代数表达式，以及总计。
(b) 为 GPT-2 XL 形状的模型实例化你的答案，得到一个只依赖于批次大小的表达式。你能使用并仍然适合 80GB 内存的最大批次大小是多少？
交付物：一个形如 𝑎 ⋅ batch_size + 𝑏 的表达式，其中 𝑎、𝑏 是数值，以及一个表示最大批次大小的数字。
(c) 运行 AdamW 的一个步骤需要多少 FLOPs？
交付物：一个代数表达式，带简要论证。
(d) 模型 FLOPs 利用率（MFU）定义为观察到的吞吐量（每秒标记数）与硬件理论峰值 FLOP 吞吐量之比 [A. Chowdhery 等人，2022]。NVIDIA H100 GPU 对"float32"（实际上是 TensorFloat-32，其实现实上是"bfloat19"）操作的理论峰值为 495 teraFLOP/s。假设你能获得 50% 的 MFU，在单个 H100 上、批次大小为 1024、训练 GPT-2 XL 400K 步需要多长时间？按照 J. Kaplan 等人 [25] 和 J. Hoffmann 等人 [26]，假设反向传播的 FLOPs 是前向传播的两倍。
交付物：训练所需的小时数，带简要论证。

### 4.4 学习率调度
导致损失最快下降的学习率值通常在训练过程中变化。在训练 Transformer 时，通常使用学习率调度，从较大的学习率开始，在开始时做出更快的更新，并随着模型训练缓慢衰减到较小的值。^8 在本作业中，我们将实现用于训练 LLaMA [H. Touvron 等人，2023] 的余弦退火调度。

调度器只是一个函数，它接受当前步数 𝑡 和其他相关参数（例如初始和最终学习率），并返回在步数 𝑡 用于梯度更新的学习率。最简单的调度是常数函数，它会对任何 𝑡 返回相同的学习率。

余弦退火学习率调度接受 (i) 当前迭代 𝑡，(ii) 最大学习率 𝛼_max，(iii) 最小（最终）学习率 𝛼_min，(iv) 预热迭代数 𝑇_𝑤，以及 (v) 余弦退火的最终迭代 𝑇_𝑐。迭代 𝑡 的学习率定义为：
- （预热）如果 𝑡 < 𝑇_𝑤，则 𝛼_𝑡 = $\frac{𝑡}{𝑇_𝑤}$ 𝛼_max。
- （余弦退火）如果 𝑇_𝑤 ≤ 𝑡 ≤ 𝑇_𝑐，则 𝛼_𝑡 = 𝛼_min + $\frac{1}{2}$(1 + cos($\frac{𝑡−𝑇_𝑤}{𝑇_𝑐−𝑇_𝑤}$𝜋))(𝛼_max − 𝛼_min)。
- （退火后）如果 𝑡 > 𝑇_𝑐，则 𝛼_𝑡 = 𝛼_min。

^8 有时常见做法是使用学习率回升（重启）的调度，以帮助越过局部最小值。

**问题（learning_rate_schedule）：实现带预热的余弦学习率调度（1 分）**
编写一个函数，接受 𝑡、𝛼_max、𝛼_min、𝑇_𝑤 和 𝑇_𝑐，并根据上述调度器返回学习率 𝛼_𝑡。然后实现 [adapters.get_lr_cosine_schedule] 并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪
在训练期间，我们有时会遇到产生大梯度的训练样本，这会破坏训练的稳定性。为了缓解这一点，实践中常用的一种技术是梯度裁剪。其思想是在每次反向传播之后、执行优化器步骤之前，对梯度的范数施加限制。给定（所有参数的）梯度 𝑔，我们计算其 ℓ_2 范数 ‖𝑔‖_2。如果该范数小于最大值 𝑀，则我们让 𝑔 保持原样；否则，我们将 𝑔 按因子 $\frac{M}{‖𝑔‖_2 + 𝜀}$ 缩小（其中为数值稳定性添加一个小的 𝜀，如 10^−6）。注意，得到的范数将略小于 𝑀。

**问题（gradient_clipping）：实现梯度裁剪（1 分）**
编写一个实现梯度裁剪的函数。你的函数应接受一个参数列表和一个最大 ℓ_2 范数。它应在原地修改每个参数的梯度。使用 𝜀 = 10^−6（PyTorch 默认值）。然后，实现适配器 [adapters.run_gradient_clipping] 并确保它通过 `uv run pytest -k test_gradient_clipping`。

## 5 训练循环
我们现在终于要把我们已经构建的主要组件放在一起：分词后的数据、模型和优化器。

### 5.1 数据加载器
分词后的数据（例如你在 tokenizer_experiments 中准备的）是单个标记序列 𝑥 = (𝑥_1, …, 𝑥_𝑛)。即使源数据可能由单独的文档组成（例如，不同的网页或源代码文件），常见做法是将所有这些连接成单个标记序列，在它们之间添加分隔符（例如 4 标记）。

数据加载器将其转换为批次流，其中每个批次由 𝐵 个长度为 𝑚 的序列组成，与对应的下一个标记配对，也长度为 𝑚。例如，对于 𝐵 = 1、𝑚 = 3，([𝑥_2, 𝑥_3, 𝑥_4], [𝑥_3, 𝑥_4, 𝑥_5]) 将是一个可能的批次。

以这种方式加载数据简化了训练，原因有很多。首先，任何 1 ≤ 𝑖 ≤ 𝑛 − 𝑚 都给出一个有效的训练序列，所以采样训练序列是平凡的。由于所有训练序列具有相同的长度，无需填充输入序列，这提高了硬件利用率（也通过增加批次大小 𝐵）。最后，我们也不需要加载整个数据集来采样训练数据，使得处理可能无法放入内存的大型数据集变得容易。

**问题（data_loading）：实现数据加载（2 分）**
交付物：编写一个函数，接受一个 numpy 数组 𝑥（整型数组，包含标记 ID）、一个 batch_size、一个 context_length 和一个 PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），并返回一对张量：采样的输入序列和相应的下一个标记目标。两个张量都应具有形状 (batch_size, context_length)，包含标记 ID，并且都应放在请求的设备上。为了对照我们提供的测试测试你的实现，你首先需要实现 [adapters.run_get_batch] 处的测试适配器。然后，运行 `uv run pytest -k test_get_batch` 来测试你的实现。

**低资源提示：CPU 或 Apple Silicon 上的数据加载**
如果你计划在 CPU 或 Apple Silicon 上训练你的 LM，你需要将数据移动到正确的设备（同样，你稍后应该为你的模型使用相同的设备）。
如果你在 CPU 上，可以使用 'cpu' 设备字符串；在 Apple Silicon（M\* 芯片）上，可以使用 'mps' 设备字符串。
有关 MPS 的更多信息，请查看以下资源：
- https://docs.pytorch.org/docs/stable/mps.html
- https://docs.pytorch.org/docs/stable/notes/mps.html
- https://developer.apple.com/documentation/metalperformanceshaders

如果数据集太大而无法加载到内存怎么办？我们可以使用名为 mmap 的 Unix 系统调用，它将磁盘上的文件映射到虚拟内存，并在访问该内存位置时惰性加载文件内容。因此，你可以"假装"整个数据集都在内存中。Numpy 通过 np.memmap（或如果你最初用 np.save 保存了数组，则使用 np.load 的 mmap_mode='r' 标志）实现这一点，它会返回一个类似 numpy 数组的对象，在你访问时按需加载条目。在训练期间从你的数据集（即 numpy 数组）采样时，务必以内存映射模式加载数据集（通过 np.memmap 或 np.load 的 mmap_mode='r' 标志，取决于你保存数组的方式）。确保你指定的 dtype 与要加载的数组匹配。显式验证内存映射的数据看起来正确（例如，不含超出预期词汇表大小的值）可能是有帮助的。

### 5.2 检查点
除了加载数据，我们还需要在训练时保存模型。当运行作业时，我们通常希望能够恢复中途停止的训练运行（例如，由于作业超时、机器故障等）。即使一切顺利，我们也可能想稍后访问中间模型（例如，事后研究训练动态、取训练不同阶段的模型样本等）。

一个检查点应包含恢复训练所需的所有状态。我们当然至少希望能够恢复模型权重。如果使用有状态优化器（如 AdamW），我们还需要保存优化器的状态（例如，对于 AdamW，是矩估计）。最后，要恢复学习率调度，我们需要知道我们停止时的迭代次数。PyTorch 使保存所有这些变得容易：每个 nn.Module 都有一个 state_dict() 方法，返回包含所有可学习权重的字典；我们稍后可以用姊妹方法 load_state_dict() 恢复这些权重。任何 torch.optim.Optimizer 也是如此。最后，torch.save(obj, dest) 可以将对象（例如，一个包含张量作为某些值的字典，但也包括整数等常规 Python 对象）转储到文件（路径）或类似文件的对象，以后可以用 torch.load(src) 加载回内存。

**问题（checkpointing）：实现模型检查点（1 分）**
实现以下两个函数来加载和保存检查点：
- `def save_checkpoint(model, optimizer, iteration, out)` 应将来自模型、优化器和迭代的所有状态转储到类似文件的对象 out 中。你可以使用模型和优化器的 state_dict 方法获取它们的相关状态，并使用 torch.save(obj, out) 将 obj 转储到 out（PyTorch 支持路径或类似文件的对象）。典型的选择是让 obj 是一个字典，但只要你以后能加载你的检查点，你可以使用任何你想要的格式。
此函数期望以下参数：
  - `model: torch.nn.Module`
  - `optimizer: torch.optim.Optimizer`
  - `iteration: int`
  - `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `def load_checkpoint(src, model, optimizer)` 应从 src（路径或类似文件的对象）加载一个检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点的迭代次数。你可以使用 torch.load(src) 恢复你在 save_checkpoint 实现中保存的内容，并在模型和优化器中使用 load_state_dict 方法将它们恢复到之前的状态。
此函数期望以下参数：
  - `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
  - `model: torch.nn.Module`
  - `optimizer: torch.optim.Optimizer`

实现 [adapters.run_save_checkpoint] 和 [adapters.run_load_checkpoint] 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环
现在，终于到了把你实现的所有组件放入你的主训练脚本的时候了。让你可以轻松地用不同超参数开始训练运行（例如，通过将它们作为命令行参数）是有回报的，因为你稍后要多次这样做，以研究不同选择如何影响训练。

**问题（training_together）：把它放在一起（4 分）**
交付物：编写一个在用户提供的输入上运行训练循环来训练你的模型的脚本。特别是，我们建议你的训练脚本至少允许以下内容：
- 能够配置和控制各种模型和优化器超参数。
- 用 np.memmap 进行大型训练和验证数据集的内存高效加载。
- 将检查点序列化到用户提供的路径。
- 定期记录训练和验证性能（例如，到控制台和/或像 Weights and Biases 这样的外部服务）。^9

^9 wandb.ai

---

## 6 生成文本
既然我们可以训练模型，我们需要的最后一块是从我们的模型生成文本的能力。回想一下，语言模型接受一个（可能批处理的）长度为 sequence_length 的整数序列，并产生一个大小为 (sequence_length, vocab_size) 的矩阵，其中序列的每个元素是一个预测该位置之后下一个标记的概率分布。我们现在将编写几个函数，将其转化为对新序列的采样方案。

**Softmax**
按照标准惯例，语言模型输出是最终线性层的输出（"logits"），所以我们必须通过 softmax 操作将其转换为归一化概率，我们之前在第 10 公式中看到过。

**解码**
为了从我们的模型生成文本（解码），我们将向模型提供一个前缀标记序列（"提示"），并要求它产生一个词汇表上的概率分布，预测序列中的下一个标记。然后，我们将从这个词汇表项目的分布中采样，以确定下一个输出标记。

具体来说，解码过程的一步应接受一个序列 𝑥_{1…𝑡}，并通过以下公式返回一个标记 𝑥_{𝑡+1}：
$$P(𝑥_{𝑡+1} = i | 𝑥_{1…𝑡}) = \frac{\exp(𝑣_i)}{\sum_j \exp(𝑣_j)} \quad (21)$$
$$v = TransformerLM(𝑥_{1…𝑡})_t ∈ ℝ^{vocab\_size} \quad (22)$$
其中 TransformerLM 是我们的模型，它接受长度为 sequence_length 的序列作为输入，并产生一个大小为 (sequence_length, vocab_size) 的矩阵，我们取这个矩阵的最后一个元素，因为我们正在寻找第 𝑡 个位置的下一标记预测。

这给了我们一个基本的解码器，通过重复从这些一步条件分布中采样（将我们之前生成的输出标记追加到下一个解码时间步的输入）直到我们生成序列结束标记 4（或用户指定的最大生成标记数）。

**解码技巧**
我们将用小型模型进行实验，而小型模型有时会生成非常低质量的文本。两个简单的解码技巧可以帮助修复这些问题。首先，在温度缩放中，我们用温度参数 𝜏 修改 softmax，其中新的 softmax 为
$$softmax(𝑣, 𝜏)_i = \frac{\exp(𝑣_i/𝜏)}{\sum_{j=1}^{vocab\_size} \exp(𝑣_j/𝜏)} \quad (23)$$
注意，设置 𝜏 → 0 会使 𝑣 的最大元素占主导地位，softmax 的输出变成一个集中于这个最大元素的一热向量。

其次，另一个技巧是核采样或 top-p 采样，我们通过截断低概率标记来修改采样分布。设 𝑞 为我们从（温度缩放的）softmax 得到的 VOCAB_SIZE 大小的概率分布。带有超参数 𝑝 的核采样根据以下公式产生下一个标记
$$P(𝑥_{𝑡+1} = i | 𝑞) = \begin{cases} \frac{𝑞_i}{\sum_{j∈𝑉^{(p)}} 𝑞_j} & \text{if } i ∈ 𝑉^{(p)} \\ 0 & \text{otherwise} \end{cases} \quad (24)$$
其中 𝑉^(p) 是满足 ∑_{j∈𝑉^(p)} 𝑞_j ≥ 𝑝 的最小索引集。你可以通过先把概率分布 𝑞 按大小排序，然后选择最大的词汇表元素直到达到目标级别 𝑝 来轻松计算这个量。

**问题（decoding）：解码（3 分）**
交付物：实现一个从你的语言模型解码的函数。我们建议你支持以下功能：
- 为用户提供的提示生成补全（即，接受一些 𝑥_{1…𝑡} 并采样一个补全，直到你遇到 4 标记）。
- 允许用户控制生成标记的最大数量。
- 给定一个期望的温度值，在采样前对预测的下一个标记分布应用 softmax 温度缩放。
- 给定用户指定的阈值，进行 Top-𝑝 采样（[A. Holtzman 等人，2020]，也称为核采样）。

## 7 实验
现在是把所有东西放在一起并在预训练数据集上训练（小型）语言模型的时候了。

### 7.1 如何运行实验和交付物
理解 Transformer 架构组件背后原理的最好方法是实际修改它并自己运行。没有什么能替代动手经验。

为此，重要的是能够快速、一致地实验，并记录你做了什么。为了快速实验，我们将在一个小规模模型（总共约 1700 万参数）和简单数据集（TinyStories）上运行许多实验。为了一致地做事情，你将系统地消融组件和变化超参数，为记录起见，我们将要求你提交实验日志和每个实验相关的学习曲线。

为了能够提交损失曲线，务必定期评估验证损失，并记录步数和墙钟时间。你可能发现像 Weights and Biases 这样的日志基础设施很有帮助。

**问题（experiment_log）：实验日志记录（3 分）**
为你的训练和评估代码创建实验跟踪基础设施，使你能够跟踪你的实验和相对于梯度步数和墙钟时间的损失曲线。
交付物：你实验的日志基础设施代码，以及本部分以下作业问题的实验日志（一份你尝试过的所有内容文档）。

### 7.2 TinyStories
我们将从一个非常简单的数据集（TinyStories；R. Eldan 等人 [1]）开始，模型会在其中快速训练，我们可以观察到一些有趣的行为。获取此数据集的说明在第 1 节。下面是一个该数据集长相的例子。

**示例（tinystories_example）：来自 TinyStories 的一个示例**
从前有一个叫 Ben 的小男孩。Ben 喜欢探索他周围的世界。他看到了许多奇妙的东西，比如商店里陈列的漂亮花瓶。一天，Ben 正在商店里走着，这时他遇到了一个非常特别的花瓶。当 Ben 看到它时，他非常惊讶！他说，"哇，那真是一个神奇的花瓶！我能买下它吗？"店主微笑着说，"当然可以。你可以把它带回家，向所有朋友展示它有多神奇！"于是 Ben 把花瓶带回家，他为它感到非常骄傲！他把朋友们叫过来，给他们看了这个神奇的花瓶。他所有的朋友都觉得这个花瓶很漂亮，不敢相信 Ben 是多么幸运。Ben 就是这样在商店里发现了一个神奇的花瓶！

#### 7.2.1 超参数调优
我们会告诉你一些非常基本的开始用的超参数，并要求你为其他一些找到好的设置。
- **词汇表大小** 10,000。典型的词汇表大小在数万到数十万之间。你应该改变它，看看词汇表和模型行为如何变化。
- **上下文长度** 256。像 TinyStories 这样简单的数据集可能不需要很长的序列长度，但对于后面的 OpenWebText 数据，你可能想要改变它。尝试改变它，看看它对每次迭代运行时和最终困惑度的影响。
- **d_model** 512。这比许多小型 Transformer 论文中使用的 768 维稍小，但这样会更快。
- **d_ff** 1344。这大约是 8/3 d_model，同时是 64 的倍数，对 GPU 性能有好处。
- **RoPE theta 参数 Θ** 10,000。
- **层数和头数** 4 层，16 头。合在一起，这将给出大约 1700 万非嵌入参数，这是一个相当小的 Transformer。
- **处理的标记总数** 327,680,000（你的批次大小 × 总步数 × 上下文长度应大约等于这个值）。

你应该通过一些试错来为以下其他超参数找到好的默认值：学习率、学习率预热、其他 AdamW 超参数（𝛽_1、𝛽_2、𝜀）和权重衰减。你可以在 D. P. Kingma 等人 [22] 中找到此类超参数的一些典型选择。

#### 7.2.2 把它放在一起
现在你可以通过得到一个训练好的 BPE 分词器、对训练数据集分词，并在你编写的训练循环中运行它来把所有东西放在一起。重要提示：如果你的实现正确且高效，上述超参数应该在 1 个 B200 GPU 上产生大约 20–30 分钟的运行时。如果你的运行时明显更长，请检查并确保你的数据加载、检查点或验证损失代码没有成为你运行时的瓶颈，并且你的实现已正确批处理。

#### 7.2.3 调试模型架构的技巧和提示
我们强烈建议熟悉你 IDE 内置的调试器（例如 VSCode/Zed），与用 print 语句调试相比，这会为你节省时间。如果你使用文本编辑器，可以使用像 ipdb 这样的工具。调试模型架构时其他一些好的做法是：
- 开发任何神经网络架构时一个常见的首要步骤是过拟合单个小批次。如果你的实现正确，你应该能够快速将训练损失驱动到接近零。
- 在各种模型组件中设置调试断点，并检查中间张量的形状，确保它们符合你的预期。
- 监控激活值、模型权重和梯度的范数，确保它们不会爆炸或消失。

**问题（learning_rate）：调优学习率（2 B200 小时）（3 分）**
学习率是最重要的调优超参数之一。基于你已训练的基础模型，回答以下问题：
(a) 对学习率进行超参数扫描，并报告最终损失（如果优化器发散，则注明发散）。
交付物：与多个学习率相关的学习曲线。解释你的超参数搜索策略。
交付物：在 TinyStories 上验证损失（每标记）最多为 1.45 的模型。

**低资源提示：在 CPU 或 Apple Silicon 上训练几个步骤**
如果你在 cpu 或 mps 上运行，则应改为将处理的标记总数减少到 40,000,000，这足以产生相当流畅的文本。你也可以将目标验证损失从 1.45 提高到 2.00。
在 M4 Max 芯片和 36 GB 内存上运行我们的解决方案代码并使用调优过的学习率，我们使用批次大小 × 总步数 × 上下文长度 = 32 × 5000 × 256 = 40,960,000 个标记，在 cpu 上需要 1 小时 22 分钟，在 mps 上需要 36 分钟。在第 5000 步，我们达到 1.80 的验证损失。
一些额外提示：
- 当使用 𝑁 个训练步骤时，我们建议调整余弦学习率衰减调度，使其衰减（即达到最小学习率）恰好终止于第 𝑁 步。
- 当使用 mps 时，不要使用 TF32 核，即不要设置 torch.set_float32_matmul_precision('high')，就像你在 cuda 设备上可能做的那样。我们尝试在 mps 上启用 TF32 核（torch 版本 2.9.0），发现后端有时使用会静默损坏、导致训练不稳定的核。
- 你可以通过用 torch.compile 对模型进行 JIT 编译来加速训练。具体来说：
  - 在 cpu 上，用以下方式编译你的模型：
    - model = torch.compile(model)
  - 在 mps 上，你可以用以下方式稍微优化反向传播：
    - model = torch.compile(model, backend="aot_eager")
  - 截至 torch 版本 2.9.0，mps 不支持用 Inductor 编译。

(b) 民间智慧说最好的学习率是"处于稳定性的边缘"。研究学习率发散的点与你的最佳学习率有何关系。
交付物：递增学习率的学习曲线，其中至少包含一个发散的运行，以及对此与收敛速度关系的分析。

现在让我们变化批次大小，看看训练会发生什么。批次大小很重要——它们让我们通过做更大的矩阵乘法从 GPU 获得更高效率，但真的我们总是想要大的批次大小吗？让我们运行一些实验来找出答案。

**问题（batch_size_experiment）：批次大小变化（1 B200 小时）（1 分）**
将你的批次大小从 1 一直变化到 GPU 内存限制。至少尝试几个中间批次大小，包括像 64 和 128 这样典型的大小。
交付物：不同批次大小运行的曲线。如有必要，学习率应再次优化。
交付物：几句讨论你对批次大小及其对训练影响的发现。

有了你的解码器在手，我们现在可以生成文本了！我们将从模型生成，看看它有多好。作为参考，你应该得到至少看起来像下面示例一样好的输出。

**示例（ts_generate_example）：来自 TinyStories 语言模型的示例输出**
从前，有一个漂亮的女孩叫 Lily。她喜欢吃口香糖，尤其是那个大的黑色口香糖。一天，Lily 的妈妈让她帮忙做晚饭。Lily 非常兴奋！她喜欢帮妈妈。Lily 的妈妈为晚餐做了一大锅汤。Lily 非常高兴，说："谢谢你，妈妈！我爱你。"她帮妈妈把汤倒进一个大碗里。晚饭后，Lily 的妈妈做了一些美味的汤。Lily 很喜欢！她说："谢谢你，妈妈！这汤太好喝了！"她的妈妈笑着说："我很高兴你喜欢，Lily。"她们做完饭后继续一起做饭。结束。

**低资源提示：在 CPU 或 Apple Silicon 上生成文本**
如果你改用了处理 40M 标记的低资源配置，你应该看到仍然像英语但不那么流畅的生成。例如，我们在 40M 标记上训练的 TinyStories 语言模型的示例输出如下：
从前，有一个叫 Sue 的小女孩。Sue 有一颗她非常喜欢的牙齿。那是他的最好的头。一天，Sue 去散步，遇到了一只瓢虫！他们成了好朋友，一起在小路上玩。
"嘿，Polly！我们出去吧！"Tim 说。Sue 看着天空，发现很难找到一种方式来跳舞闪闪发光。她微笑着同意帮助那说话的！"
当 Sue 看着天空移动，它是什么。她

下面是精确的问题陈述和我们的要求：
**问题（generate）：生成文本（1 分）**
使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要操作解码器参数（温度、top-p 等）以获得流畅的输出。
交付物：至少 256 个标记的文本转储（或直到第一个 4 标记），以及对这个输出流畅性的简要评论，以及至少两个影响这个输出好坏的因素。

### 7.3 消融和架构修改
理解 Transformer 的最好方法是实际修改它并观察它的行为。我们现在将做一些简单的消融和修改。

**消融 1：层归一化**
人们常说层归一化对 Transformer 训练的稳定性很重要。但也许我们想冒险。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

**问题（layer_norm_ablation）：移除 RMSNorm 并训练（0.5 B200 小时）（1 分）**
从你的 Transformer 中移除所有 RMSNorm 并训练。在之前最优的学习率下会发生什么？你能通过使用更低的学习率获得稳定性吗？
交付物：当你移除 RMSNorm 并训练时的学习曲线，以及最佳学习率的学习曲线。
交付物：几句关于 RMSNorm 影响的评论。

现在让我们研究另一个乍一看似乎任意的层归一化选择。Pre-norm Transformer 块定义为
$$𝑧 = 𝑥 + MultiHeadSelfAttention(RMSNorm(𝑥)) \quad (25)$$
$$𝑦 = 𝑧 + FFN(RMSNorm(𝑧))。 \quad (26)$$
这是对原始 Transformer 架构的少数"共识"修改之一，原始架构使用 post-norm 方法
$$𝑧 = RMSNorm(𝑥 + MultiHeadSelfAttention(𝑥)) \quad (27)$$
$$𝑦 = RMSNorm(𝑧 + FFN(𝑧))。 \quad (28)$$
让我们恢复到 post-norm 方法，看看会发生什么。

**问题（pre_norm_ablation）：实现 post-norm 并训练（0.5 B200 小时）（1 分）**
将你的 pre-norm Transformer 实现修改为 post-norm。用 post-norm 模型训练，看看会发生什么。
交付物：post-norm Transformer 的学习曲线，与 pre-norm 的学习曲线比较。

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

**消融 2：位置嵌入**
接下来我们将研究位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（带 RoPE）与完全不包括位置嵌入（NoPE）。事实证明，仅解码器 Transformer，即那些像我们实现的那样带有因果掩码的，理论上可以在没有显式提供位置嵌入的情况下推断相对或绝对位置信息 [Y.-H. H. Tsai 等人，2019；A. Kazemnejad 等人，2023]。我们现在将经验性地测试 NoPE 与 RoPE 相比表现如何。

**问题（no_pos_emb）：实现 NoPE（0.5 B200 小时）（1 分）**
修改你的带 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。
交付物：比较 RoPE 和 NoPE 性能的学习曲线。

**消融 3：SwiGLU 与 SiLU**
接下来，我们将遵循 N. Shazeer [20]，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但不使用门控线性单元（GLU）的前馈网络的性能，来测试门控在前馈网络中的重要性：
$$FFN_{SiLU}(𝑥) = 𝑊_2 SiLU(𝑊_1 𝑥)。 \quad (29)$$
回想一下，在我们的 SwiGLU 实现中，我们将内部前馈层的维度设置为大约 d_ff = 8/3 d_model（同时确保 d_ff mod 64 = 0，以利用 GPU 张量核心）。在这个消融基线中，你的 FFN_SiLU 实现应改为设置 d_ff = 4 × d_model，以大致匹配默认 SwiGLU 前馈网络的参数数量（它有三个而不是两个权重矩阵）。

**问题（swiglu_ablation）：SwiGLU 与 SiLU（0.5 B200 小时）（1 分）**
交付物：比较 SwiGLU 和 SiLU 前馈网络性能的学习曲线，参数数量大致匹配。
交付物：几句讨论你的发现。

**低资源提示：GPU 资源有限的在线学生应在 TinyStories 上测试修改**
在作业的剩余部分，我们将转向一个更大规模、更嘈杂的网络数据集（OpenWebText），用架构修改进行实验，（可选）提交到课程排行榜。
在 OpenWebText 上把 LM 训练到流畅需要很长时间，所以我们建议 GPU 有限的在线学生继续在 TinyStories 上测试修改（用验证损失作为评估性能的指标）。

### 7.4 在 OpenWebText 上运行
我们现在将转向一个从网络爬虫创建的更标准的预训练数据集。OpenWebText [A. Gokaslan 等人，2019] 的一个小样本也作为单个文本文件提供：参见第 1 节了解如何访问此文件。

这里是 OpenWebText 的一个示例。注意文本更加真实、复杂和多样化。你可能想浏览训练数据集，以了解网络爬取语料库的训练数据长什么样。

**示例（owt_example）：来自 OWT 的一个示例**
Baseball Prospectus 的技术总监 Harry Pavlidis 在雇佣 Jonathan Judge 时冒了一个风险。
Pavlidis 知道，正如 Alan Schwarz 在《数字游戏》中所写，"美国文化中没有哪个角落比棒球运动员的表现被更精确地计数、更热情地量化。"只需几下点击，你就能发现 Noah Syndergaard 的快球在飞向本垒板的路上每分钟旋转超过 2,100 次，Nelson Cruz 在 2016 年合格击球手中拥有联盟最高的平均击出初速，还有无数其他似乎从电子游戏或科幻小说中撕下来的花絮。不断上升的数据洪流赋权了棒球文化中一个日益重要的角色：分析爱好者。
这种赋权伴随着额外的审视——对测量本身，也对它们背后的人和出版物。对于 Baseball Prospectus，Pavlidis 了如指掌伴随量化不完美的强烈反对。他也知道该网站的捕手指标需要重新设计，而这需要一个有学问的头脑——一个能处理复杂统计建模问题的人——来完成这项工作。
"他让我们害怕。"Harry Pavlidis
Pavlidis 有一个预感，Judge"懂行"，基于后者的写作和他们在网站主办的球场活动中的互动。[…]

注意：你可能需要为此实验重新调整你的超参数，如学习率或批次大小。

**问题（main_experiment）：在 OWT 上实验（2 B200 小时）（2 分）**
在 OpenWebText 上训练你的语言模型，使用与 TinyStories 相同的模型架构和总训练迭代次数。这个模型做得怎么样？
交付物：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 相比损失的差异——我们应该如何解释这些损失？
交付物：来自 OpenWebText LM 的生成文本，格式与 TinyStories 输出相同。这段文本的流畅度如何？为什么即使我们拥有与 TinyStories 相同的模型和计算预算，输出质量却更差？

### 7.5 你自己的修改 + 排行榜
祝贺你走到这一步。你几乎完成了！你现在将尝试改进 Transformer 架构，看看你的超参数和架构与班上的其他学生相比如何。

**排行榜规则**
除了以下限制外没有其他限制：
- **运行时：** 你的提交最多可以在 B200 上运行 45 分钟。如果你使用 SLURM 或 Modal，你可能想在提交脚本中强制执行这一点。
- **数据：** 你只能使用我们提供的 OpenWebText 训练数据集。
否则，你可以自由地为所欲为。

如果你在寻找一些实现想法，你可以查看以下一些资源：
- 最先进的开源 LLM 系列，如 Llama 3 [A. Grattafiori 等人，2024] 或 Qwen 2.5 [A. Yang 等人，2024]。
- NanoGPT 速通仓库（github.com/KellerJordan/modded-nanogpt），社区成员在那里发布许多用于"速通"小规模语言模型预训练的有趣修改。例如，一个可以追溯到原始 Transformer 论文的常见修改是将输入和输出嵌入的权重绑定在一起（参见 A. Vaswani 等人 [8]（第 3.4 节）和 A. Chowdhery 等人 [16]（第 2 节））。如果你确实尝试权重绑定，你可能需要降低嵌入/LM head 初始化的标准差。

在尝试完整的 45 分钟运行之前，你要在 OpenWebText 的小子集或 TinyStories 上测试这些。

作为警告，我们确实注意到你可能在此排行榜中发现有效的一些修改可能不会泛化到更大规模的预训练。我们将在课程的缩放定律单元中进一步探讨这个想法。

**问题（leaderboard）：排行榜（10 B200 小时）（6 分）**
你将按照上述排行榜规则训练一个模型，目标是在 0.75 B200 小时内最小化你语言模型的验证损失。
交付物：最终记录的验证损失，一个清楚显示小于 45 分钟墙钟时间 x 轴的相关学习曲线，以及你做了什么描述。我们期望排行榜提交至少击败 5.0 损失的朴素基线。在此提交到排行榜：github.com/stanford-cs336/assignment1-basics-leaderboard。

## 参考文献
[1] R. Eldan 和 Y. Li，"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" 2023.
[2] A. Gokaslan, V. Cohen, E. Pavlick, 和 S. Tellex，"OpenWebText corpus." 2019.
[3] R. Sennrich, B. Haddow, 和 A. Birch，"Neural Machine Translation of Rare Words with Subword Units," in Proc. of ACL, 2016.
[4] C. Wang, K. Cho, 和 J. Gu，"Neural Machine Translation with Byte-Level Subwords." 2019.
[5] P. Gage，"A new algorithm for data compression," C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.
[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, 和 I. Sutskever，"Language Models are Unsupervised Multitask Learners." 2019.
[7] A. Radford, K. Narasimhan, T. Salimans, 和 I. Sutskever，"Improving Language Understanding by Generative Pre-Training." 2018.
[8] A. Vaswani 等人，"Attention is All you Need," in Proc. of NeurIPS, 2017.
[9] T. Q. Nguyen 和 J. Salazar，"Transformers without Tears: Improving the Normalization of Self-Attention," in Proc. of IWSWLT, 2019.
[10] R. Xiong 等人，"On Layer Normalization in the Transformer Architecture," in Proc. of ICML, 2020.
[11] J. L. Ba, J. R. Kiros, 和 G. E. Hinton，"Layer Normalization." 2016.
[12] H. Touvron 等人，"LLaMA: Open and Efficient Foundation Language Models." 2023.
[13] B. Zhang 和 R. Sennrich，"Root Mean Square Layer Normalization," in Proc. of NeurIPS, 2019.
[14] A. Grattafiori 等人，"The Llama 3 Herd of Models." [Online]. Available: https://arxiv.org/abs/2407.21783
[15] A. Yang 等人，"Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.
[16] A. Chowdhery 等人，"PaLM: Scaling Language Modeling with Pathways." 2022.
[17] D. Hendrycks 和 K. Gimpel，"Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.
[18] S. Elfwing, E. Uchibe, 和 K. Doya，"Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." [Online]. Available: https://arxiv.org/abs/1702.03118
[19] Y. N. Dauphin, A. Fan, M. Auli, 和 D. Grangier，"Language Modeling with Gated Convolutional Networks." [Online]. Available: https://arxiv.org/abs/1612.08083
[20] N. Shazeer，"GLU Variants Improve Transformer." 2020.
[21] J. Su, Y. Lu, S. Pan, B. Wen, 和 Y. Liu，"RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
[22] D. P. Kingma 和 J. Ba，"Adam: A Method for Stochastic Optimization," in Proc. of ICLR, 2015.
[23] I. Loshchilov 和 F. Hutter，"Decoupled Weight Decay Regularization," in Proc. of ICLR, 2019.
[24] T. B. Brown 等人，"Language Models are Few-Shot Learners," in Proc. of NeurIPS, 2020.
[25] J. Kaplan 等人，"Scaling Laws for Neural Language Models." 2020.
[26] J. Hoffmann 等人，"Training Compute-Optimal Large Language Models." 2022.
[27] A. Holtzman, J. Buys, L. Du, M. Forbes, 和 Y. Choi，"The Curious Case of Neural Text Degeneration," in Proc. of ICLR, 2020.
[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, 和 R. Salakhutdinov，"Transformer Dissection: An Unified Understanding for Transformer`\'s Attention via the Lens of Kernel," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, 和 X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.
[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, 和 S. Reddy，"The Impact of Positional Encoding on Length Generalization in Transformers," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=Drrl2gcjzl

---

*本翻译为教学辅助材料，翻译可能不完全贴合原文措辞，请以英文原文为准。*