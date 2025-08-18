from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/tongyi/protgpt/protgpt3")

# 测试序列
sample_seq = "EVQLVESGGGLVQPGGSLRLSCAAS"  # 示例抗体序列片段
encoded = tokenizer(sample_seq, return_tensors="pt")
decoded = tokenizer.batch_decode(encoded["input_ids"])

# 打印结果
print("输入序列:", sample_seq)
print("分词后序列:", decoded)
print("是否包含 <|endoftext|>:", "<|endoftext|>" in decoded[0])