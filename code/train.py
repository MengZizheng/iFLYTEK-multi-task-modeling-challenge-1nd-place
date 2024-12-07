from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from transformers import AutoTokenizer
import nest_asyncio

nest_asyncio.apply()


pipe = pipeline("/root/onethingai-tmp/user_data/LLaMA-Factory/models/qwen2_5_lora_sft")
tokenizer = AutoTokenizer.from_pretrained("/root/onethingai-tmp/user_data/LLaMA-Factory/models/qwen2_5_lora_sft", trust_remote_code=True)

import pandas as pd 
import json 
import re


df_train = pd.read_csv("../xfdata/train.csv", sep="\t")
df_test = pd.read_csv("../xfdata/test_submit.csv", sep="\t")

def split_into_sentences(sentence):
    # 使用正则表达式根据句号、问号、感叹号进行分割
    sentence_list = re.split(r'(?<=[。！？])', sentence)
    # 去掉空字符串和前后空格
    sentence_list = [s.strip() for s in sentence_list if s.strip()]
    return sentence_list

from tqdm import tqdm 


result = []
for content in tqdm(df_test["content"]):
    try:
        # print(f"==================================================[{idx}|{len(df_test)}]==================================================")
        result_ = []
        # 实体抽取（人名提取）
        sentences = split_into_sentences(content)
        person = []
        for sentence in sentences:
            conversation = [
                {
                    "role": "system",
                    "content": "请基于以下内容，完成实体抽取（人名提取）。"
                },
                {
                    "role": "user",
                    "content": sentence
                }
            ]
            response = pipe(tokenizer.decode(tokenizer.apply_chat_template(conversation)) + " \n").text
            # print(f"sentence: {sentence}")
            # print(f"person: {response}")
            if response != "无":
                person.extend(response.split(","))
        name_list = [i for i in set(person) if i in content]
        # 获取人名的首次出现位置
        name_positions = [(name, content.find(name)) for name in name_list if content.find(name) != -1]
        # 根据出现位置排序
        sorted_names = sorted(name_positions, key=lambda x: x[1])
        # 提取排序后的名字列表
        sorted_name_list = [name for name, _ in sorted_names]
        person = ",".join(sorted_name_list)
        if person == "无":
            person = np.NaN
        result_.append(person)
    
        # 情感分析（正向或负向）
        # response = model.chat(tokenizer, content, [], meta_instruction="请基于以下内容，完成情感分析（正向或负向）。")
        conversation = [
                {
                    "role": "system",
                    "content": "请基于以下内容，完成情感分析（正向或负向）。"
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        response = pipe(tokenizer.decode(tokenizer.apply_chat_template(conversation)) + " \n").text
        # print(f"content: {content}")
        # print(f"sentiment: {response}")
        result_.append(response)
    
        # 文本翻译（中文翻译为英文）
        # response = model.chat(tokenizer, content, [], meta_instruction="请基于以下内容，完成文本翻译（中文翻译为英文）")
        trans = ""
        for sentence in sentences:
            conversation = [
                {
                    "role": "system",
                    "content": "请基于以下内容，完成文本翻译（中文翻译为英文）"
                },
                {
                    "role": "user",
                    "content": sentence
                }
            ]
            response = pipe(tokenizer.decode(tokenizer.apply_chat_template(conversation)) + " \n").text
            if response[-1] == " ":
                trans += response
            else:
                trans += response + " "
        trans = trans.strip()
        result_.append(trans)
        print(result_)
        result.append(result_)
    except:
        result.append(["", "", ""])

df_test[["person", "sentiment", "en"]] = result
df_test.to_csv("../prediction_result/submit.csv", sep="\t", index=False)