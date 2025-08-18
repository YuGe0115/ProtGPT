from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 离线加载模型和分词器
model_path = "/home/tongyi/protgpt/protgpt3"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 验证模型加载
print("模型加载成功:", model.config.model_type)