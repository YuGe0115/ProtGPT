from datasets import load_dataset
from transformers import AutoTokenizer

# 加载数据并添加 EOS 标记
def add_eos(example):
    example["text"] = example["text"] + "<|endoftext|>"
    return example

dataset = load_dataset("text", data_files={"train": "/home/tongyi/protgpt/antibody.txt"})
dataset = dataset.map(add_eos)

# 分割数据集（80%训练，20%验证）
dataset = dataset["train"].train_test_split(test_size=0.2)

# 离线加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/tongyi/protgpt/protgpt3")

# Tokenize函数
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

# 应用tokenize
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 保存数据集
tokenized_dataset.save_to_disk("/home/tongyi/protgpt/tokenized_antibody_dataset")

print("数据集准备完成:", tokenized_dataset)