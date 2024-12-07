from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

import pandas as pd 

df_train = pd.read_csv("../xfdata/train.csv", sep="\t")
df_test = pd.read_csv("../xfdata/test_submit.csv", sep="\t")

import json 
import re

def split_into_sentences(sentence):
    # 使用正则表达式根据句号、问号、感叹号进行分割
    sentence_list = re.split(r'(?<=[。！？])', sentence)
    # 去掉空字符串和前后空格
    sentence_list = [s.strip() for s in sentence_list if s.strip()]
    return sentence_list
    

multi_task = []
for content, person, sentiment, en in df_train.values:
    # 实体抽取（人名提取）
    if str(person) != "nan":
        person = person.split(",")
        sentences = split_into_sentences(content)
        for sentence in sentences:
            name_list = [i for i in person if i in sentence]
            # 获取人名的首次出现位置
            name_positions = [(name, sentence.find(name)) for name in name_list if sentence.find(name) != -1]
            # 根据出现位置排序
            sorted_names = sorted(name_positions, key=lambda x: x[1])
            # 提取排序后的名字列表
            sorted_name_list = [name for name, _ in sorted_names]
            output = ",".join(sorted_name_list)
            multi_task.append({
                "instruction": "请基于以下内容，完成实体抽取（人名提取）。",
                "input": sentence,
                "output": output if output else "无"
            })
    # 情感分析（正向或负向）
    multi_task.append({
        "instruction": "请基于以下内容，完成情感分析（正向或负向）。",
        "input": content,
        "output": sentiment
    })
    # 文本翻译（中文翻译为英文）
    multi_task.append({
        "instruction": "请基于以下内容，完成文本翻译（中文翻译为英文）",
        "input": content,
        "output": en
    })

with open("../user_data/LLaMA-Factory/data/multi_task.json", "w", encoding="utf-8") as f:
    json.dump(multi_task, f, indent=4, ensure_ascii=False)