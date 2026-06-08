import torch
import numpy as np
import os
import tiktoken


class BinaryDataset:
    def __init__(self, data_dir='.', block_size=256, device='cpu'):
        self.block_size = block_size
        self.device = device

        train_path = os.path.join(data_dir, 'train.bin')
        val_path = os.path.join(data_dir, 'val.bin')

        if not os.path.exists(train_path):
            raise FileNotFoundError("请先运行 prepare_data.py 生成二进制数据文件！")

        # Memmap 模式: 数据留在硬盘，按需读取到内存
        self.train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        self.val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

        self.enc = tiktoken.get_encoding("gpt2")

    def get_batch(self, split='train', batch_size=32):
        data = self.train_data if split == 'train' else self.val_data

        # 随机采样起始点
        ix = torch.randint(len(data) - self.block_size, (batch_size,))

        # 转为 int64 Tensor (PyTorch 需要 Long 类型)
        x = torch.stack([torch.from_numpy((data[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + self.block_size + 1]).astype(np.int64)) for i in ix])

        if self.device != 'cpu':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.enc.decode(ids)