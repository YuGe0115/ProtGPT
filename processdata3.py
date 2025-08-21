import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

# 加载ProtGPT2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/tongyi/protgpt/protgpt3")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

# 读取TXT文件
def read_sequences(file_path):
    with open(file_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]  # 移除空行和换行符
    return sequences

sequences = read_sequences("/mnt/ssd3/tongyi/antibody.txt")  # 你的TXT文件路径
print(f"读取序列数: {len(sequences)}") 


# 划分数据集
train_seqs, temp_seqs = train_test_split(sequences, test_size=0.2)
val_seqs, test_seqs = train_test_split(temp_seqs, test_size=0.5)
print(f"训练集: {len(train_seqs)}, 验证集: {len(val_seqs)}, 测试集: {len(test_seqs)}")

# 创建自定义PyTorch Dataset
class ProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=256):  # max_length根据你的序列长度设置，ProtGPT2支持到1024
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx] + '<|endoftext|>'  # 手动添加 EOS
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

    # def __getitem__(self, idx):
        # seq = self.sequences[idx]
        # # Tokenize并添加EOS
        # encoding = self.tokenizer(
        #     seq,
        #     add_special_tokens=True,  # 自动添加EOS (ProtGPT2的默认行为)
        #     max_length=self.max_length,
        #     truncation=True,
        #     return_tensors="pt"
        # )
        # input_ids = encoding['input_ids'].squeeze(0)  # 移除batch维度
        # attention_mask = encoding['attention_mask'].squeeze(0)
        # return {'input_ids': input_ids, 'attention_mask': attention_mask}

# 创建Dataset实例
train_dataset = ProteinDataset(train_seqs, tokenizer)
val_dataset = ProteinDataset(val_seqs, tokenizer)
test_dataset = ProteinDataset(test_seqs, tokenizer)

# 加载模型并调整词汇表大小
model = AutoModelForCausalLM.from_pretrained("/home/tongyi/protgpt/protgpt3")
# 由于添加了新的 pad_token，需调整模型的嵌入层大小
model.resize_token_embeddings(len(tokenizer))  # 同步词汇表大小
model.config.pad_token_id = tokenizer.pad_token_id  # 设置模型的 pad_token_id

# 自定义collate_fn处理padding和labels
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]

    # 动态padding到batch内最大长度
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # 对于生成模型，labels是input_ids的右移版本（shift right），忽略padding
    labels = input_ids_padded.clone()
    labels[labels == tokenizer.pad_token_id] = -100  # 忽略padding的loss计算

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels
    }

# 创建DataLoader
batch_size = 5  # 根据GPU内存调整，小数据量建议小batch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)



















# 迭代一个batch查看格式（Torch可识别的Tensor）
# for batch in train_loader:
#     print("Batch 示例:")
#     print("input_ids shape:", batch['input_ids'].shape)  # e.g., torch.Size([8, max_len_in_batch])
#     print("attention_mask shape:", batch['attention_mask'].shape)
#     print("labels shape:", batch['labels'].shape)
#     break  # 只看第一个batch



















# 验证 padding 情况
# print("检查 padding 情况：")
# for batch in train_loader:
#     print("Batch 示例:")
#     print("input_ids shape:", batch['input_ids'].shape)  # 形状：[batch_size, max_len_in_batch]
#     print("attention_mask shape:", batch['attention_mask'].shape)
#     print("labels shape:", batch['labels'].shape)

#     # 打印第一个序列的 input_ids 和 attention_mask
#     print("\n第一个序列的 input_ids:", batch['input_ids'][0])
#     print("第一个序列的 attention_mask:", batch['attention_mask'][0])
#     print("第一个序列的 labels:", batch['labels'][0])

#     # 解码 input_ids，检查序列和 [PAD] token
#     decoded_seq = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
#     print("解码后的第一个序列:", decoded_seq)

#     # 检查 padding token 的出现
#     pad_token_id = tokenizer.pad_token_id
#     print(f"pad_token_id: {pad_token_id}")
#     pad_count = (batch['input_ids'] == pad_token_id).sum().item()
#     print(f"batch 中 [PAD] token 总数: {pad_count}")

#     # 统计 batch 中每个序列的实际长度（基于 attention_mask）
#     seq_lengths = batch['attention_mask'].sum(dim=1).tolist()
#     print("batch 中各序列的实际长度（不含 padding）:", seq_lengths)

#     break  # 只检查第一个 batch









# 验证 padding 和 EOS
# print("验证 padding 和 EOS：")
# for batch in train_loader:
#     print("Batch 示例:")
#     print("input_ids shape:", batch['input_ids'].shape)
#     print("attention_mask shape:", batch['attention_mask'].shape)
#     print("labels shape:", batch['labels'].shape)

#     # 获取 pad_token_id 和 eos_token_id
#     pad_token_id = tokenizer.pad_token_id
#     eos_token_id = tokenizer.eos_token_id
#     print(f"pad_token_id: {pad_token_id}, eos_token: {eos_token_id}")

#     # 检查第一个序列
#     for i in range(min(2, batch_size)):  # 检查 batch 中前 2 个序列
#         input_ids = batch['input_ids'][i]
#         attention_mask = batch['attention_mask'][i]
#         labels = batch['labels'][i]

#         # 解码序列，包含特殊 token
#         decoded_seq = tokenizer.decode(input_ids, skip_special_tokens=False)
#         print(f"\n序列 {i+1} 解码:", decoded_seq)

#         # 检查 padding：padding 部分应为 pad_token_id
#         padding_positions = input_ids == pad_token_id
#         if padding_positions.any():
#             print(f"序列 {i+1} 有 padding，位置: {padding_positions.nonzero(as_tuple=True)[0].tolist()}")
#             print(f"padding 部分是否为 [PAD]: {(input_ids[padding_positions] == pad_token_id).all().item()}")
#             print(f"padding 部分是否不用 EOS: {(input_ids[padding_positions] != eos_token_id).all().item()}")

#         # 检查 EOS：有效序列末尾（最后一个 1 后的 token）应为 eos_token_id
#         valid_length = attention_mask.sum().item()  # 有效长度
#         if valid_length > 1:  # 确保序列非空
#             last_valid_token = input_ids[valid_length - 1]
#             print(f"序列 {i+1} 有效部分末尾 token: {last_valid_token}, 是否为 EOS: {last_valid_token == eos_token_id}")

#         # 验证 labels 的 padding 部分
#         print(f"序列 {i+1} labels padding 是否为 -100: {(labels[padding_positions] == -100).all().item()}")

#     break  # 只检查第一个 batch





























from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
import pandas as pd

# 配置训练参数
training_args = TrainingArguments(
    output_dir="/mnt/ssd3/tongyi/finetuned",
    num_train_epochs=20,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=293,
    save_steps=293,
    learning_rate=1e-5,
    save_total_limit=None,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    logging_dir="/mnt/ssd3/tongyi/finetuned/logs",
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# 训练
trainer.train()

# 保存模型和 tokenizer
model.save_pretrained("/mnt/ssd3/tongyi/finetuned/final_model")
tokenizer.save_pretrained("/mnt/ssd3/tongyi/finetuned/final_model")
print("模型已保存到 /mnt/ssd3/tongyi/finetuned/final_model")

# 提取 log 数据
log_history = trainer.state.log_history
train_loss = [log['loss'] for log in log_history if 'loss' in log]
eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
steps = [log['step'] for log in log_history if 'loss' in log or 'eval_loss' in log][:len(train_loss + eval_loss)]

# 保存 log 到 CSV
log_df = pd.DataFrame({
    'step': [log['step'] for log in log_history],
    'train_loss': [log.get('loss', None) for log in log_history],
    'eval_loss': [log.get('eval_loss', None) for log in log_history]
})
log_df.to_csv("/mnt/ssd3/tongyi/finetuned/logs/training_log.csv", index=False)
print("Log 已保存到 /mnt/ssd3/tongyi/finetuned/logs/training_log.csv")

# 绘制 loss 曲线
plt.plot([s for s, t in zip(steps, train_loss) if t], train_loss, label="Train Loss")
plt.plot([s for s, e in zip(steps, eval_loss) if e], eval_loss, label="Eval Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("/mnt/ssd3/tongyi/finetuned/loss_curve.png")
plt.show()
print("Loss 曲线已保存到 /mnt/ssd3/tongyi/finetuned/loss_curve.png")
