from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
import torch
import matplotlib.pyplot as plt

# 离线加载模型和分词器
model_path = "/home/tongyi/protgpt/protgpt3"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").to("cuda:3")  # 指定 GPU 3
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 加载 tokenized 数据集
tokenized_dataset = load_from_disk("/home/tongyi/protgpt/tokenized_antibody_dataset")

# 数据收集器（因果语言建模）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir="/home/tongyi/protgpt/finetuned_protgpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,  # 3 个 epoch，适合 1800 条数据
    per_device_train_batch_size=4,  # 小 batch 避免 OOM
    per_device_eval_batch_size=4,
    eval_strategy="epoch",  # 每个 epoch 评估
    save_steps=200,  # 每 200 步保存检查点
    save_total_limit=2,  # 保留最近 2 个检查点
    learning_rate=5e-5,  # 小学习率避免灾难性遗忘
    weight_decay=0.01,
    fp16=True,  # 混合精度加速（需 GPU）
    push_to_hub=False,  # 离线保存
    logging_dir="/home/tongyi/protgpt/finetuned_protgpt2_logs",
    logging_steps=100,  # 每 100 步记录日志
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

# 保存最终模型和分词器
trainer.save_model("/home/tongyi/protgpt/finetuned_protgpt2_final")
tokenizer.save_pretrained("/home/tongyi/protgpt/finetuned_protgpt2_final")
print("微调完成，模型已保存至 /home/tongyi/protgpt/finetuned_protgpt2_final")

print("log_history:", log_history)
print("train_loss:", train_loss)
print("eval_loss:", eval_loss)

# 绘制 loss 曲线
# 提取训练过程中的日志历史，包含 train_loss 和 eval_loss
log_history = trainer.state.log_history
# 从日志中提取训练 loss（每 logging_steps 记录一次）
train_loss = [log['loss'] for log in log_history if 'loss' in log]
# 从日志中提取验证 loss（每个 epoch 记录一次）
eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
# 创建训练 loss 的步数（与 train_loss 长度一致）
steps = range(len(train_loss))
# 绘制训练 loss 曲线
plt.plot(steps, train_loss, label="Train Loss")
# 如果存在验证 loss，绘制验证 loss 曲线（步数对应评估点）
if eval_loss:
    eval_steps = [i for i, log in enumerate(log_history) if 'eval_loss' in log]
    plt.plot(eval_steps, eval_loss, label="Eval Loss")
# 设置 x 轴标签为“Steps”
plt.xlabel("Steps")
# 设置 y 轴标签为“Loss”
plt.ylabel("Loss")
# 设置图表标题
plt.title("Training and Evaluation Loss Curve")
# 添加图例，区分训练和验证 loss
plt.legend()
# 保存 loss 曲线为 PNG 文件
plt.savefig("/home/tongyi/protgpt/loss_curve.png")
# 关闭图表，释放内存
plt.close()
# 打印保存路径，方便检查
print("Loss 曲线已保存至 /home/tongyi/protgpt/loss_curve.png")