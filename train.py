import torch
import time
from config import LlamaConfig
from dataset import BinaryDataset
from model import Llama


def train():
    # 1. 设置
    cfg = LlamaConfig()
    print(f"Device: {cfg.device}")

    # 2. 数据
    try:
        dataset = BinaryDataset(data_dir='.', block_size=cfg.max_seq_len, device=cfg.device)
    except FileNotFoundError:
        print("错误: 未找到数据文件，请先运行 'python prepare_data.py'")
        return

    # 3. 模型
    model = Llama(cfg).to(cfg.device)
    # 启用 torch.compile 可以加速 (Linux Only, Windows请注释掉)
    # model = torch.compile(model)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # 5. Loop
    print("开始训练...")
    t0 = time.time()
    model.train()

    for i in range(cfg.max_iters):
        # 获取 Batch
        xb, yb = dataset.get_batch('train', cfg.batch_size)

        # Forward & Backward
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 日志
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.4f}")

        # 验证集评估
        if i % cfg.eval_iters == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = dataset.get_batch('val', cfg.batch_size)
                _, vloss = model(vx, vy)
                print(f"--- Eval: Val Loss {vloss.item():.4f} ---")
            model.train()

    print(f"训练完成，耗时: {time.time() - t0:.2f}s")

    # 保存模型
    torch.save(model.state_dict(), "baby_llama3.pth")

    # 6. 生成测试 (Text Completion)
    print("\n=== 生成测试 ===")
    model.eval()

    # 提示词: "Once upon a time" (编码后)
    prompt_text = "Once upon a time"
    prompt_ids = dataset.enc.encode(prompt_text)
    context = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)

    generated_ids = model.generate(context, max_new_tokens=100)
    print(dataset.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    train()