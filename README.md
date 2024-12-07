# 大模型多任务建模挑战赛方案

## 赛题背景

本次挑战赛要求参赛者使用开源大模型进行自然语言多任务建模，任务包括：

- **实体抽取**（人名提取）
- **情感分析**（正向或负向）
- **文本翻译**（中文翻译为英文）

赛题链接：[大模型多任务建模挑战赛](https://challenge.xfyun.cn/topic/info?type=multi-task-modeling-challenge)

数据链接：[比赛数据集](https://challenge.xfyun.cn/topic/info?type=multi-task-modeling-challenge&option=stsj)

## 方案概述

我们采用 **Qwen2.5-7B-Instruct** 作为基础大模型，并使用 `LLaMA-Factory` 框架对其进行微调。最终，我们实现了多任务建模，包括实体抽取、情感分析和文本翻译。

## 关键代码部分

### 1. 数据预处理

在数据预处理阶段，我们提取每个任务的输入并格式化为模型能够处理的格式。以下是对每个任务的处理方式：

#### 实体抽取（人名提取）

```python
def split_into_sentences(sentence):
    sentence_list = re.split(r'(?<=[。！？])', sentence)
    sentence_list = [s.strip() for s in sentence_list if s.strip()]
    return sentence_list

# 遍历训练集中的每一条数据
for content, person, sentiment, en in df_train.values:
    if str(person) != "nan":
        person = person.split(",")
        sentences = split_into_sentences(content)
        for sentence in sentences:
            # 提取人名
            name_list = [i for i in person if i in sentence]
            name_positions = [(name, sentence.find(name)) for name in name_list if sentence.find(name) != -1]
            sorted_names = sorted(name_positions, key=lambda x: x[1])
            sorted_name_list = [name for name, _ in sorted_names]
            output = ",".join(sorted_name_list)
            multi_task.append({
                "instruction": "请基于以下内容，完成实体抽取（人名提取）。",
                "input": sentence,
                "output": output if output else "无"
            })
```

#### 情感分析

```python
multi_task.append({
    "instruction": "请基于以下内容，完成情感分析（正向或负向）。",
    "input": content,
    "output": sentiment
})
```

#### 文本翻译

```python
multi_task.append({
    "instruction": "请基于以下内容，完成文本翻译（中文翻译为英文）",
    "input": content,
    "output": en
})
```

### 2. 微调过程

使用 `LLaMA-Factory` 进行微调训练。以下是训练流程的关键命令：

```bash
# 运行数据预处理脚本
python preprocess.py

# 切换到 LLaMA-Factory 目录进行模型训练
cd ../user_data/user/LLaMA-Factory

# 训练微调模型
llamafactory-cli train examples/train_lora/qwen2_5_lora_sft.yaml

# 导出训练后的模型
llamafactory-cli export examples/merge_lora/qwen2_5_lora_sft.yaml
```

### 3. 预测过程

使用微调后的模型进行测试集的预测。每个测试样本进行人名提取、情感分析和文本翻译。

```python
pipe = pipeline("/root/onethingai-tmp/user_data/LLaMA-Factory/models/qwen2_5_lora_sft")
tokenizer = AutoTokenizer.from_pretrained("/root/onethingai-tmp/user_data/LLaMA-Factory/models/qwen2_5_lora_sft", trust_remote_code=True)

# 对每一条测试数据进行预测
for content in tqdm(df_test["content"]):
    result_ = []
    
    # 实体抽取
    response = pipe(tokenizer.decode(tokenizer.apply_chat_template(conversation)) + " \n").text
    result_.append(response)
    
    # 情感分析
    response = pipe(tokenizer.decode(tokenizer.apply_chat_template(conversation)) + " \n").text
    result_.append(response)
    
    # 文本翻译
    trans = ""
    for sentence in sentences:
        response = pipe(tokenizer.decode(tokenizer.apply_chat_template(conversation)) + " \n").text
        trans += response + " "
    result_.append(trans.strip())

    result.append(result_)
```

## 依赖环境

安装所需的 Python 库：

```bash
pip install transformers lmdeploy tqdm pandas
```

## 总结

本方案通过微调开源大模型 **Qwen2.5-7B-Instruct**，成功完成了多任务建模，预测结果准确地涵盖了实体抽取、情感分析和翻译任务。

## 算力平台

为了高效训练模型，我们使用了 [onethingai](https://onethingai.com/invitation?code=wGZHFckZ) 提供的算力平台。该平台提供了强大的GPU资源，使我们能够在较短的时间内完成模型训练和微调。

## 贡献者

- **团队名称**：小老正
- **成员**：[孟子正]
