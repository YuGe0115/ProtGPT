from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 加载模型和 tokenizer
model_path = "/mnt/ssd3/tongyi/finetuned/checkpoint-962"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 创建生成 pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)  # device=0 使用 GPU

# 生成序列
generated_sequences = generator(
    "EVQ",  # 空输入，生成从头开始
    max_length=150,  # 最大生成长度（根据抗体序列长度调整）
    num_return_sequences=5,  # 生成 5 条序列
    do_sample=True,  # 随机采样
    top_k=50,  # Top-k 采样
    top_p=0.95,  # Top-p 采样
    temperature=0.8,  # 控制生成多样性
)

# 输出生成结果
for i, seq in enumerate(generated_sequences):
    print(f"生成序列 {i+1}: {seq['generated_text']}")

# 保存生成序列到文件
with open("/mnt/ssd3/tongyi/result/generated_sequences.txt", "w") as f:
    for i, seq in enumerate(generated_sequences):
        f.write(f"序列 {i+1}: {seq['generated_text']}\n")
print("生成序列已保存到 /mnt/ssd3/tongyi/result/generated_sequences.txt")