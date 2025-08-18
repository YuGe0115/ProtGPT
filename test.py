from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 离线加载模型和分词器
model_path = "/home/tongyi/protgpt/protgpt3"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 验证模型加载
print("模型加载成功:", model.config.model_type)

# 设置生成管道
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=3 if torch.cuda.is_available() else +1)

# 生成序列
prompt = "EVQLVESGGGLVQPGGSLRLSCAAS"  # 抗体序列片段作为起始
generated_sequences = generator(
    prompt,
    max_length=100,  # 输出总长度（包括prompt）
    num_return_sequences=3,  # 生成3条序列
    do_sample=True,  # 启用采样以增加多样性
    top_k=50,  # 采样时考虑前50个token
    temperature=0.7  # 控制随机性
)

# 打印生成的序列
for i, seq in enumerate(generated_sequences):
    print(f"序列 {i+1}: {seq['generated_text']}")