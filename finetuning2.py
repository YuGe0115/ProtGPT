from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
import torch
import matplotlib.pyplot as plt

# 离线加载模型和分词器
model_path = "/home/tongyi/protgpt/protgpt3"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 设置 padding token 为 <|endoftext|>，确保与 prepare_data.py 一致
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("已设置 pad_token 为 eos_token: ", tokenizer.pad_token)

# 加载 tokenized 数据集
tokenized_dataset = load_from_disk("/home/tongyi/protgpt/tokenized_antibody_dataset")
print("训练集大小:", len(tokenized_dataset["train"]))
print("验证集大小:", len(tokenized_dataset["validation"]))
print("测试集大小:", len(tokenized_dataset["test"]))
for i in range(min(5, len(tokenized_dataset["train"]))):
    print(f"训练集序列 {i+1} input_ids 长度: {len(tokenized_dataset['train'][i]['input_ids'])}")

# 数据收集器（因果语言建模）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir="/home/tongyi/protgpt/finetuned_protgpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_steps=200,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
    logging_dir="/home/tongyi/protgpt/finetuned_protgpt2_logs",
    logging_steps=10,
    n_gpu=3,
    max_steps=-1,
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 开始微调
trainer.train()
print("总步数:", trainer.state.global_step)
print("处理的样本数:", trainer.state.global_step * training_args.per_device_train_batch_size)

# 保存最终模型和分词器
trainer.save_model("/home/tongyi/protgpt/finetuned_protgpt2_final")
tokenizer.save_pretrained("/home/tongyi/protgpt/finetuned_protgpt2_final")
print("微调完成，模型已保存至 /home/tongyi/protgpt/finetuned_protgpt2_final")

# 绘制 loss 曲线
try:
    log_history = trainer.state.log_history
    print("log_history:", log_history)
    train_loss = [log['loss'] for log in log_history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    print("train_loss:", train_loss)
    print("eval_loss:", eval_loss)
    steps = range(len(train_loss))
    plt.plot(steps, train_loss, label="Train Loss")
    if eval_loss:
        eval_steps = [i for i, log in enumerate(log_history) if 'eval_loss' in log]
        plt.plot(eval_steps, eval_loss, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.savefig("/home/tongyi/protgpt/loss_curve.png")
    plt.close()
    print("Loss 曲线已保存至 /home/tongyi/protgpt/loss_curve.png")
except AttributeError as e:
    print(f"错误: {e}. 请检查 trainer.train() 是否成功运行并生成日志。")