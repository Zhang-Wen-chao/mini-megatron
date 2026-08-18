# Benchmarks

> mini-megatron 的所有性能数据怎么读、怎么复现、怎么扩展。

> 实验口径：本页的历史表格是 legacy evidence。新的性能结论必须遵守
> [可信实验协议](experiment-protocol.md)：配置等价、至少五个配对重复、保存
> manifest 与原始 Nsight 产物。

---

## 零、最新受控证据（2026-08-18，clean tree）

最新 L20 会话已在已提交源码 (`ad82d7e`)、GPU 空闲预检下完成 5 组交替配对，
所有 run bundle 校验和与两份 Nsight 原始产物均已验证：

- **公平 TP=1 FP32 对比**：同权重转换、同一 230 个固定 next-token batch、同一
  无 bias GPT 合同、标准 AdamW、5 个交替配对，mini/MCore 配对均值为
  **1.179204x**（范围 1.178079-1.180554；mini 32,669.6，MCore 27,704.8 tok/s）。
  每个 manifest 均为 `source_tree_clean=true`。此结果只适用于该精确的
  单卡 FP32 合同，不能外推为通用框架结论。
- 旧的 mini fused 对仓库 Megatron-Core custom-loop baseline 2.289x 测量没有共享
  权重、固定输入或相同前向图，现仅保留为脚本端到端开销的诊断记录，**不可解读为
  框架性能胜负**。
- mini fused 对 mini unfused：**+17.08%**（1.1708x 配对均值；5 组）。
- 正确的 L20 容器完整测试：**38 passed in 11.28s**。

完整环境、逐项统计、原始 .nsys-rep/SQLite 路径与 SHA-256 见
[2026-08-17/18 evidence ledger](experiment-results-2026-08-17.md)。旧的
dirty-tree 样本只作为历史对照保留，不参与当前汇总。
该 custom-loop baseline 不构成“mini-megatron 普遍快于 Megatron”的主张；重跑前
必须完成同权重、同输入、同语义的一步校验。
BF16 目前没有通过同一数值门槛，不能与 FP32 公平结论混写。

## 一、历史数据（2026-08-11）：fused AdamW

### 50 步 benchmark（4×L20 48GB, BF16）

| 配置 | BF16 (--amp) | BF16 + fused (--amp --fused) | gain |
|------|--------------|------------------------------|------|
| **TP=1 PP=1** | 51,700 tok/s, 36.18% MFU | **60,617 tok/s, 42.42% MFU** | **+17.2%** |
| **TP=2 PP=1** | 26,133 tok/s, 9.14% MFU | 28,126 tok/s, 9.84% MFU | +7.6% |
| **TP=2 PP=2** | 23,186 tok/s, 4.21% MFU | 23,261 tok/s, 4.22% MFU | +0.3% |

> 测量方法：unfused/fused 交替各跑 2 轮（每轮 50 测量步 + 10 warmup，同 seed=42 随机数据），取稳定轮次。`--fused` 把 AdamW 优化器步骤合并为单 kernel（Nsight Systems 测量：优化器 kernel 时间从 45.2% 降到 26.3%，占 wall-clock 从 38.5% 降到 26.3%）。单卡 125M 时收益最大（优化器是内存带宽瓶颈）；TP/PP 切分后每 rank 参数变少，优化器不再主导，收益趋近于零。
> 训练等价性：固定数据文件 + 同 seed 下 unfused/fused 逐 step loss 对比，50 步 diff=0，200 步 Max diff ~1e-4（非位级一致，浮点归约顺序差异）。
> 完整测试条件：`docs/nsight-adamw-optimizer-bottleneck.md` 附录

---

## 二、历史数据（2026-07-24）

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

## 三、怎么读这些数据

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

## 四、复现数据

### 受控 benchmark（替代手工 50 步样本）

```bash
python3 experiments/run_experiment.py --name mini-125m-tp1-fused-r01 \
  --tag variant=mini --tag pair=01 --tag condition=125m-s512-b4-bf16-fused-200x30 -- \
  torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 --num-steps 200 \
  --warmup-steps 30 --micro-batch-size 4 --amp --fused
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
# 每个命令独立生成一个 run bundle。按 ABBA/BAAB 顺序运行五对，再汇总：
python3 experiments/summarize_paired_results.py --results-dir results/runs \
  --left mini --right megatron --output results/aggregates/125m-tp1.json
```

---

## 五、怎么保存新数据

跑完新 benchmark 后：

```bash
# 1. 先验证每个 bundle，再生成 aggregate（不能挑最快样本）
python3 experiments/validate_run_bundle.py results/runs/<run-id>

# 2. 保存 manifest、环境、命令、CSV、checksums 和分析 JSON；
#    .nsys-rep/.sqlite 留在不可变归档或 Git LFS，并记录 URI、大小、SHA-256。

# 3. 只有 clean-tree、至少五个配对和语义等价检查完成后，才更新 README 摘要。
```

---

## 六、添加新 benchmark

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

## 七、已知问题

- Megatron-Core baseline 在 BF16 模式下吞吐和 FP32 一样（`16,408 vs 16,479 tok/s`），因为 `Float16Module` 包装在小模型上开销 > 收益
- 2000 步对比中 Megatron 到 2000 步 loss=0.30 还没完全收敛，再跑 5000 步可能更好
- 7B+ 模型的 benchmark 没跑过（125M 是测试过的上限）
- Linux x86 + NVIDIA GPU 是测试过的环境；Apple Silicon (MPS) 没测

---

## 八、性能调优指南

发现 mini-megatron 慢时按这个顺序排查：

1. **用 BF16**（`--amp`）—— 1.1-1.5x 免费
2. **检查 batch size** —— 小 batch 浪费 SM
3. **检查 seq_len** —— 太长 OOM，太短浪费 attention
4. **profile** —— `python -c "import torch; torch.profiler.profile(...)"` 找瓶颈
5. **减少 Python 循环** —— Python 端是瓶颈就 vectorize
6. **考虑 F.scaled_dot_product_attention** —— 已用，自动选最优

参考实现里的优化（`comm/overlap_*.py` 等）只在 ≥1B 模型有收益。
