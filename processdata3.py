import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

# 加载ProtGPT2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/tongyi/protgpt/protgpt3")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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
        seq = self.sequences[idx]
        # Tokenize并添加EOS
        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,  # 自动添加EOS (ProtGPT2的默认行为)
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0)  # 移除batch维度
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

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

# 示例: 迭代一个batch查看格式（Torch可识别的Tensor）
for batch in train_loader:
    print("Batch 示例:")
    print("input_ids shape:", batch['input_ids'].shape)  # e.g., torch.Size([8, max_len_in_batch])
    print("attention_mask shape:", batch['attention_mask'].shape)
    print("labels shape:", batch['labels'].shape)
    break  # 只看第一个batch