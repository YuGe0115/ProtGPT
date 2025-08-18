from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载微调模型和分词器
model_path = "/home/tongyi/protgpt/finetuned_protgpt2_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

# 设置 padding token 和 eos token，与微调一致
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
    print("已设置 pad_token: ", tokenizer.pad_token)
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
    print("已设置 eos_token: ", tokenizer.eos_token)

# 多 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU")
    model = torch.nn.DataParallel(model, device_ids=[3, 2])  # 使用 GPU 3 和 2
else:
    model = model.to(device)
model.eval()

# 生成参数
generate_kwargs = {
    "max_length": 150,  # 与 prepare_data.py 的 max_length 一致
    "num_return_sequences": 10,  # 生成 10 条序列
    "do_sample": True,  # 随机采样
    "top_k": 50,  # Top-k 采样
    "top_p": 0.95,  # Top-p 采样
    "temperature": 1.0,  # 控制随机性
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# 输入提示（可选，空输入从头生成）
prompt = ""  # 可设为抗体序列片段，例如 "EVQLV"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 生成序列
with torch.no_grad():
    outputs = model.module.generate(**inputs, **generate_kwargs) if isinstance(model, torch.nn.DataParallel) else model.generate(**inputs, **generate_kwargs)

# 解码生成结果
sequences = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]

# 打印并保存序列
output_file = "/home/tongyi/protgpt/generated_sequences.txt"
with open(output_file, "w") as f:
    for i, seq in enumerate(sequences):
        print(f"序列 {i+1}: {seq[:50]}...")  # 打印前 50 个字符
        f.write(seq + "\n")

print(f"生成序列已保存至 {output_file}")