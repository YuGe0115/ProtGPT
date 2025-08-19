sequences = []
num_sequences = 10
for i in range(num_sequences):
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
        seq = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # 后处理：去除连续 <|endoftext|>
        seq = re.sub(r"(<\|endoftext\|>)+", "<|endoftext|>", seq)
        sequences.append(seq)
        print(f"序列 {i+1}: {seq[:50]}... (长度: {len(seq) - len('<|endoftext|>')})")