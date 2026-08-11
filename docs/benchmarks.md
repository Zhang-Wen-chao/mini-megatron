# Benchmarks

> mini-megatron 的所有性能数据怎么读、怎么复现、怎么扩展。

---

## 零、最新数据（2026-08-11）：fused AdamW

### 50 步 benchmark（4×L20 48GB, BF16）

| 配置 | BF16 (--amp) | BF16 + fused (--amp --fused) | gain |
|------|--------------|------------------------------|------|
| **TP=1 PP=1** | 45,371 tok/s, 31.75% MFU | **60,625 tok/s, 42.42% MFU** | **+33.6%** |
| **TP=2 PP=1** | 21,126 tok/s, 7.39% MFU | 22,775 tok/s, 7.96% MFU | +7.8% |
| **TP=2 PP=2** | 23,229 tok/s, 5.70% MFU | 23,530 tok/s, 5.77% MFU | +1.3% |

> `--fused` 把 AdamW 优化器步骤合并为单 kernel（Nsight Systems 测量：优化器 kernel 时间从 45.2% 降到 26.3%）。单卡 125M 时收益最大（优化器是内存带宽瓶颈）；TP/PP 切分后每 rank 参数变少，优化器不再主导，收益趋近于零。
> 完整分析：`docs/nsight-adamw-optimizer-bottleneck.md`

---

## 一、历史数据（2026-07-24）

### 50 步快速 benchmark（4×L20 48GB）

| 配置 | mini-megatron |  | Megatron-Core |  |
|------|---------------|--|---------------|--|
| | FP32 | BF16 | | FP32 | BF16 |
| **TP=1 PP=1** | 34,152 tok/s, 23.90% MFU | 38,873 tok/s, 27.20% MFU | 16,479 tok/s, 11.53% MFU | 16,408 tok/s, 11.48% MFU |
| **TP=2 PP=1** | 31,048 tok/s, 10.86% MFU | 47,126 tok/s, 16.49% MFU | 19,471 tok/s, 6.80% MFU | 19,250 tok/s, 6.74% MFU |
| **TP=2 PP=2** | 31,699 tok/s, 5.62% MFU | 32,191 tok/s, 5.70% MFU | - | - |

**mini-megatron 加速**：1.6-2.4x（FP32 1.59-2.07x，BF16 2.37x）

### 2000 步训练对比（identity 任务）

| 步数 | mini-megatron | Megatron-Core | mini 更快 |
|------|---------------|---------------|-----------|
| 200 | 0.06 | 9.88 | 165x |
| 1000 | 0.0074 | 2.50 | 338x |
| 2000 | **0.0054** | **0.30** | **55x** |

两个最终都收敛到接近 0，但 mini 快 55x（2000 步时）。

完整数据：`results/identity_2000steps.json`

---

## 二、怎么读这些数据

### "tok/s" 是什么
每秒处理的 token 数。`B × S × num_steps / elapsed`

### "MFU" 是什么
Model FLOPs Utilization，实际算力 / 理论峰值。

公式（来自 `main.py:compute_mfu`）：
```
attn_proj = 24 × H × H            # attention 投影
mlp = 48 × H × H                  # FFN
logits = 6 × H × V                # lm head
flops_per_step = (attn + mlp) × L × tokens + 6 × L × H × S × S
mfu = flops_per_step × num_steps × dp_w / (elapsed × 110e12 × gpu_world)
```

`110e12` 是 L20 FP16 matmul 峰值（~110 TFLOPS）。

### 为什么 mini-megatron 快

| 因素 | 解释 |
|------|------|
| 无 DDP 包装 | Megatron 用 `DistributedDataParallel`，对单卡/小模型有 wrapper 开销 |
| 无 Float16Module | Megatron 强加 FP16 包装，125M 模型反而拖慢 |
| `torch.autocast` | 按需转换，无 wrapper 开销 |
| 直白训练循环 | ~50 行，无 framework 抽象 |

### 为什么 mini-megatron 收敛快（2000 步）

| 因素 | 解释 |
|------|------|
| 无 scaled output init | Megatron `output_layer_init_method` 用 `0.02/sqrt(2L)≈0.0041`（GPT-2 设计），前 1500 步输出层几乎不动 |
| 无 FusedLayerNorm | Megatron 的 Triton 实现对小模型 gradient 路径不同 |
| F.scaled_dot_product_attention | 自动选最优 kernel，无 wrapper |

---

## 三、复现数据

### 50 步 benchmark

```bash
cd <repo-root>

# mini-megatron TP=1 PP=1
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 --num-steps 50

# mini-megatron TP=1 PP=1 BF16
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 --num-steps 50 --amp

# Megatron-Core TP=1 PP=1
torchrun --nproc_per_node=1 eval/run_megatron_baseline.py --tp 1 --pp 1 --num-steps 50
```

### 50 步 benchmark（fused AdamW）

```bash
# mini-megatron TP=1 PP=1 BF16 + fused AdamW
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 --num-steps 50 --amp --fused
```

### 2000 步对比

```bash
cd <repo-root>

# 1. 生成数据
python3 experiments/synthetic_data.py experiments/synthetic_data.pt
python3 experiments/make_identity.py    # 自动基于 synthetic_data.pt 生成

# 2. 跑两个框架
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
  --num-steps 2000 --warmup-steps 50 \
  --data-file experiments/identity_data.pt
torchrun --nproc_per_node=1 eval/run_megatron_baseline.py --tp 1 --pp 1 \
  --num-steps 2000 --warmup-steps 50 \
  --data-file experiments/identity_data.pt

# 3. 用 compare_convergence.py 一键对比
python3 experiments/compare_convergence.py --data-file experiments/identity_data.pt
```

### 自动化所有 benchmark

```bash
# TODO: 一键跑所有 benchmark
# 当前需要手动跑每条
```

---

## 四、怎么保存新数据

跑完新 benchmark 后：

```bash
# 1. 把数据更新到 results/identity_2000steps.json
#    格式在 tests/test_identity_results.py 里有

# 2. 更新 README 表格

# 3. 在 benchmarks.md（本文档）加新的测试条件

# 4. 提交
git add results/ README.md dev-guides/mini-megatron/benchmarks.md
git commit -m "docs: update benchmark with new test conditions"
git push origin main
```

---

## 五、添加新 benchmark

需要测试新场景（如 7B 模型、不同 batch size、新硬件）时：

1. **写跑脚本**（放 `experiments/`）
2. **输出结构化数据**（json / csv）
3. **加测试**（在 `tests/test_*_results.py`）
4. **更新本文档**（新的测试条件、新数据）
5. **更新 README**（性能表格）

### 示例：加 7B 模型 benchmark

```bash
# 1. 在 config.py 加 7B 配置
# 2. 跑 benchmark
torchrun --nproc_per_node=4 main.py --tp 2 --pp 2 --num-steps 100
# 3. 解析输出存为 results/7b_tp2pp2.json
# 4. 在 tests/test_7b_results.py 加验证
# 5. 在 README 和 benchmarks.md 加新表
```

---

## 六、已知问题

- Megatron-Core baseline 在 BF16 模式下吞吐和 FP32 一样（`16,408 vs 16,479 tok/s`），因为 `Float16Module` 包装在小模型上开销 > 收益
- 2000 步对比中 Megatron 到 2000 步 loss=0.30 还没完全收敛，再跑 5000 步可能更好
- 7B+ 模型的 benchmark 没跑过（125M 是测试过的上限）
- Linux x86 + NVIDIA GPU 是测试过的环境；Apple Silicon (MPS) 没测

---

## 七、性能调优指南

发现 mini-megatron 慢时按这个顺序排查：

1. **用 BF16**（`--amp`）—— 1.1-1.5x 免费
2. **检查 batch size** —— 小 batch 浪费 SM
3. **检查 seq_len** —— 太长 OOM，太短浪费 attention
4. **profile** —— `python -c "import torch; torch.profiler.profile(...)"` 找瓶颈
5. **减少 Python 循环** —— Python 端是瓶颈就 vectorize
6. **考虑 F.scaled_dot_product_attention** —— 已用，自动选最优

参考实现里的优化（`comm/overlap_*.py` 等）只在 ≥1B 模型有收益。
