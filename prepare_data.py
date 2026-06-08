import os
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# === 配置 ===
# 为了演示快速运行，我们只取前 20,000 个故事
# 如果想训练更好的模型，可以把 NUM_SAMPLES 设为 None (全量数据)
NUM_SAMPLES = 20000
VAL_RATIO = 0.1


def prepare():
    print("正在加载 TinyStories 数据集...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # 收集数据
    data = []
    print(f"提取前 {NUM_SAMPLES} 个故事...")
    for i, item in tqdm(enumerate(dataset), total=NUM_SAMPLES):
        if i >= NUM_SAMPLES: break
        data.append(item['text'])

    # 划分训练/验证集
    n = len(data)
    train_data = data[:int(n * (1 - VAL_RATIO))]
    val_data = data[int(n * (1 - VAL_RATIO)):]

    # 使用 tiktoken (GPT-2 BPE)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # <|endoftext|>

    def process(texts, filename):
        # 预分配列表
        ids = []
        for text in tqdm(texts, desc=f"Tokenizing {filename}"):
            # 编码文本 + 添加结束符
            ids.extend(enc.encode_ordinary(text))
            ids.append(eot)

        # 转换为 uint16 (节省空间, gpt2 词表 < 65535)
        ids = np.array(ids, dtype=np.uint16)
        ids.tofile(filename)
        print(f"已保存 {filename}, Token总数: {len(ids) / 1e6:.2f}M")

    process(train_data, "train.bin")
    process(val_data, "val.bin")
    print(f"数据处理完成！词表大小: {enc.n_vocab}")


if __name__ == "__main__":
    prepare()