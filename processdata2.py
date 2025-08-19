from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import DatasetDict

# 加载数据
dataset = load_dataset("text", data_files={"train": "/home/tongyi/protgpt/antibody.txt"})
# 统计序列长度以优化 max_length
lengths = [len(example["text"]) for example in dataset["train"]]
max_length = min(max(lengths), 200)  # 限制 max_length 为 200，适合抗体序列
print(f"最大序列长度: {max_length}, 平均序列长度: {sum(lengths)/len(lengths):.2f}")

# 分割数据集：70% 训练，15% 验证，15% 测试
train_validtest = dataset["train"].train_test_split(test_size=0.3)
valid_test = train_validtest["test"].train_test_split(test_size=0.5)

tokenizer = AutoTokenizer.from_pretrained("/home/tongyi/protgpt/protgpt3")
if "<PAD>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
tokenizer.pad_token = "<PAD>"

# 离线加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/tongyi/protgpt/protgpt3")
tokenizer.pad_token = "<PAD>"  # 自定义填充 token

# 创建 DatasetDict
dataset = DatasetDict({
    "train": train_validtest["train"],
    "validation": valid_test["train"],
    "test": valid_test["test"]
})

# Tokenize 函数
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length",add_special_tokens=True)

# 应用 tokenize
tokenized_dataset = DatasetDict({
    "train": dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"]),
    "validation": dataset["validation"].map(tokenize_function, batched=True, remove_columns=["text"]),
    "test": dataset["test"].map(tokenize_function, batched=True, remove_columns=["text"])
})

for i in range(min(5, len(tokenized_dataset["train"]))):
    input_ids = tokenized_dataset["train"][i]["input_ids"]
    print(f"训练集序列 {i+1} input_ids 长度: {len(input_ids)}")
    print(f"解码序列: {tokenizer.decode(input_ids)}")

# 验证数据集内容
for i in range(min(5, len(tokenized_dataset["train"]))):
    print(f"训练集序列 {i+1} input_ids 长度: {len(tokenized_dataset['train'][i]['input_ids'])}")

# 保存数据集
tokenized_dataset.save_to_disk("/home/tongyi/protgpt/tokenized_antibody_dataset2")

print("数据集准备完成:", tokenized_dataset)
print("训练集大小:", len(tokenized_dataset["train"]))
print("验证集大小:", len(tokenized_dataset["validation"]))
print("测试集大小:", len(tokenized_dataset["test"]))