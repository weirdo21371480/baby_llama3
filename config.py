from dataclasses import dataclass
import torch


@dataclass
class LlamaConfig:
    # --- 模型架构参数 ---
    dim: int = 288  # 嵌入维度
    n_layers: int = 6  # 层数
    n_heads: int = 6  # Query 头数
    n_kv_heads: int = 3  # Key/Value 头数 (GQA: kv < heads)
    vocab_size: int = 50304  # 填充到 128 的倍数 (GPT-2原大小是50257)
    multiple_of: int = 32  # FFN 维度对齐
    norm_eps: float = 1e-5
    max_seq_len: int = 256  # 故事上下文长度
    dropout: float = 0.1

    # --- 训练参数 ---
    batch_size: int = 32
    learning_rate: float = 5e-4
    max_iters: int = 5000  # 迭代次数
    eval_iters: int = 200  # 每隔多少步评估一次
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'