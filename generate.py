from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载微调模型和分词器
model_path = "/home/tongyi/protgpt/finetuned_protgpt2_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

# 设置 padding token 和 eos token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
    print("已设置 pad_token: ", tokenizer.pad_token)
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
    print("已设置 eos_token: ", tokenizer.eos_token)

# 设置多 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 主 GPU 为 cuda:0
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # 映射到 cuda:0, cuda:1
model = model.to(device)  # 确保模型在主 GPU
model.eval()

# 生成参数
generate_kwargs = {
    "max_length": 150,
    "num_return_sequences": 10,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 1.0,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# 输入提示
prompt = ""
inputs = tokenizer(prompt, return_tensors="pt").to(device)  # 确保输入在主 GPU (cuda:3)

# 生成序列
with torch.no_grad():
    outputs = model.module.generate(**inputs, **generate_kwargs) if isinstance(model, torch.nn.DataParallel) else model.generate(**inputs, **generate_kwargs)

# 解码生成结果
sequences = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]

# 打印并保存序列
output_file = "/home/tongyi/protgpt/generated_sequences.txt"
with open(output_file, "w") as f:
    for i, seq in enumerate(sequences):
        print(f"序列 {i+1}: {seq[:50]}...")
        f.write(seq + "\n")

print(f"生成序列已保存至 {output_file}")