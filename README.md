<div align="center">

# 🦙 Baby-Llama3

**从零复现 Llama 3 架构 | 纯 PyTorch · 无高层 API**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-TinyStories-orange?style=flat)](https://huggingface.co/datasets/roneneldan/TinyStories)

</div>

---

## 📖 项目简介

**Baby-Llama3** 是一个教育向的 Llama 3 底层复现项目——**每一行代码手写，每一个算子从零实现**。不依赖 `nn.Transformer`、不调用 Flash Attention 库、不使用 HuggingFace 模型。目标是透彻理解现代大语言模型的核心机制。

适用于：深度学习进阶、LLM 底层原理学习、PyTorch 实战练习。

## 🏗️ 架构总览

```
Input Tokens
     │
     ▼
┌────────────┐
│  Embedding  │
└────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│           Transformer Block × N         │
│  ┌──────────────────────────────────┐   │
│  │       RMSNorm (Pre-Norm)         │   │
│  │            ↓                      │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │   GQA (Grouped Query Attn) │  │   │
│  │  │   · RoPE 旋转位置编码      │  │   │
│  │  │   · Causal Mask            │  │   │
│  │  └────────────────────────────┘  │   │
│  │            ↓  (+ Residual)       │   │
│  │       RMSNorm (Pre-Norm)         │   │
│  │            ↓                      │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │ FeedForward (SwiGLU)       │  │   │
│  │  │   SiLU ⊗ Linear            │  │   │
│  │  └────────────────────────────┘  │   │
│  │            ↓  (+ Residual)       │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
     │
     ▼
┌────────────┐
│  RMSNorm   │
└────────────┘
     │
     ▼
┌────────────┐
│  LM Head   │ ──▶ Logits / Loss
└────────────┘
```

## ✨ 核心特性

| 模块 | 实现要点 |
|------|----------|
| **RMSNorm** | Pre-Norm 归一化，手写 `rsqrt` 数值稳定版 |
| **RoPE** | 复数域旋转位置编码，`precompute_freqs_cis` 缓存加速 |
| **GQA** | 分组查询注意力，K/V 头数 < Q 头数，`repeat_interleave` 广播 |
| **Causal Mask** | 自回归因果遮罩，`torch.tril` 下三角矩阵 |
| **SwiGLU** | SiLU 门控 + 线性投影，`8/3 · dim` 隐藏层缩放 |
| **Weight Init** | `N(0, 0.02)` 正态初始化，Embedding/Linear 统一 |

### Llama 3 vs GPT-2 关键区别

| | GPT-2 | Llama 3 |
|---|-------|---------|
| 归一化 | LayerNorm (Post-Norm) | **RMSNorm (Pre-Norm)** |
| 位置编码 | Learned Positional Embedding | **RoPE** |
| 注意力 | Multi-Head Attention (MHA) | **Grouped Query Attention (GQA)** |
| 激活函数 | GELU | **SwiGLU** |
| Q/K/V 偏差 | 有 bias | **无 bias** |

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.10
PyTorch >= 2.0
tiktoken
datasets
tqdm
```

### 安装依赖

```bash
pip install torch tiktoken datasets tqdm numpy
```

### 三步运行

```bash
# Step 1: 数据预处理（下载 TinyStories + BPE 编码 → .bin）
python prepare_data.py

# Step 2: 训练（~5K iterations, CPU/GPU 自适应）
python train.py

# Step 3: 查看生成结果（训练完成后自动生成故事续写）
```

## ⚙️ 模型配置

```python
@dataclass
class LlamaConfig:
    dim: int = 288           # 隐藏维度
    n_layers: int = 6        # Transformer 层数
    n_heads: int = 6         # Query 注意力头数
    n_kv_heads: int = 3      # Key/Value 注意力头数（GQA）
    vocab_size: int = 50304  # 词表大小（对齐 128 倍数）
    max_seq_len: int = 256   # 最大上下文长度
    dropout: float = 0.1     # Dropout 比例
    batch_size: int = 32     # 批次大小
    learning_rate: float = 5e-4
    max_iters: int = 5000    # 训练迭代次数
```

**参数量：~1.8M**，适合单卡/CPU 训练，数分钟即可看到效果。

## 📊 训练流程

```
Epoch Loop
  ├── 随机采样 Batch (Memmap 零拷贝读取)
  ├── Forward: Logits + Cross-Entropy Loss
  ├── Backward: AdamW Optimizer
  ├── ── Logging (每 100 steps)
  └── ── Validation (每 200 steps)
```

- **优化器**: AdamW（`lr=5e-4`）
- **数据加载**: `np.memmap` 内存映射，支持大规模数据低内存训练
- **混合精度**: 可配合 `torch.amp` 加速
- **评估策略**: 训练集 Loss + 验证集 Loss 双曲线监控

## 📁 项目结构

```
baby_llama3/
├── config.py          # 模型 & 训练超参数
├── model.py           # Llama 3 核心架构
│   ├── RMSNorm        # Root Mean Square Normalization
│   ├── RoPE           # Rotary Position Embedding
│   ├── GQA Attention  # Grouped Query Attention
│   ├── SwiGLU FFN     # Swish-Gated Linear Unit
│   └── Llama.generate # 自回归文本生成
├── dataset.py         # Memmap 数据加载器
├── prepare_data.py    # TinyStories → BPE Tokenizer → .bin
├── train.py           # 训练入口 + 生成演示
└── README.md
```

## 🎯 生成示例

```
Prompt: "Once upon a time"
────────────────────────────────────
Once upon a time, there was a little
girl named Lily who loved to play
in the garden. She saw a butterfly
and decided to follow it...
```

## 📚 参考资料

- [Llama 3 论文](https://arxiv.org/abs/2407.21783) — Meta 官方技术报告
- [RoPE 论文](https://arxiv.org/abs/2104.09864) — Rotary Position Embeddings
- [TinyStories](https://arxiv.org/abs/2305.07759) — 小模型训练数据集
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — Andrej Karpathy 的纯 C Llama 实现

---

<div align="center">
  <sub>Built with ❤️ for LLM learners</sub>
</div>
