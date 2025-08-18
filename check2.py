from datasets import load_from_disk
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/tongyi/protgpt/protgpt3")

# 加载数据集
tokenized_dataset = load_from_disk("/home/tongyi/protgpt/tokenized_antibody_dataset")

# 检查训练集前几个序列
print("检查训练集序列：")
for i in range(min(5, len(tokenized_dataset["train"]))):  # 查看前5条
    input_ids = tokenized_dataset["train"][i]["input_ids"]
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"序列 {i+1}: {decoded}")
    print(f"是否包含 <|endoftext|>: {'<|endoftext|>' in decoded}")

# 检查验证集（可选）
print("\n检查验证集序列：")
for i in range(min(5, len(tokenized_dataset["test"]))):
    input_ids = tokenized_dataset["test"][i]["input_ids"]
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"序列 {i+1}: {decoded}")
    print(f"是否包含 <|endoftext|>: {'<|endoftext|>' in decoded}")