from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import DatasetDict

# 加载数据并添加 EOS 标记
def add_eos(example):
    example["text"] = example["text"] + "<|endoftext|>"
    return example

dataset = load_dataset("text", data_files={"train": "/home/tongyi/protgpt/antibody.txt"})
dataset = dataset.map(add_eos)

# 分割数据集：70% 训练，15% 验证，15% 测试
train_validtest = dataset["train"].train_test_split(test_size=0.3)  # 先分出 70% 训练 + 30% 其他
valid_test = train_validtest["test"].train_test_split(test_size=0.5)  # 30% 均分为验证和测试

# 离线加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/tongyi/protgpt/protgpt3")

# 创建 DatasetDict
dataset = DatasetDict({
    "train": train_validtest["train"],      # 70% ≈ 1260 条
    "validation": valid_test["train"],      # 15% ≈ 270 条
    "test": valid_test["test"]              # 15% ≈ 270 条
})

# Tokenize函数
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

# 应用 tokenize
tokenized_dataset = DatasetDict({
    "train": dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"]),
    "validation": dataset["validation"].map(tokenize_function, batched=True, remove_columns=["text"]),
    "test": dataset["test"].map(tokenize_function, batched=True, remove_columns=["text"])
})

# 保存数据集
tokenized_dataset.save_to_disk("/home/tongyi/protgpt/tokenized_antibody_dataset")

print("数据集准备完成:", tokenized_dataset)

# 验证数据集大小
print("训练集大小:", len(tokenized_dataset["train"]))
print("验证集大小:", len(tokenized_dataset["validation"]))
print("测试集大小:", len(tokenized_dataset["test"]))