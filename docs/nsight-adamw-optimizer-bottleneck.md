# 用 Nsight Systems 找出小模型训练里被忽视的 45%：优化器才是隐藏瓶颈

> 工具实践 | mini-megatron 性能调优记录
> 环境：4×NVIDIA L20 (48GB, GDDR6 @ 864GB/s) · NGC PyTorch 26.01 · 单卡 125M GPT · BF16

## 背景

最近在给面试做准备（方向：AI 框架/分布式训练性能优化），把个人项目 mini-megatron（800 行纯 PyTorch 实现的 Megatron 并行策略）拿出来做一次正经的性能剖析。基线数据（单卡 125M，BF16）：

- **吞吐 51,700 tok/s，MFU 36.2%**
- 对比 Megatron-Core 同配置 11.48% MFU，已经快 1.6-2.4 倍
- 但对照 NVIDIA 参考（H100 47%），36.2% 仍有提升空间

任务：用 NVIDIA 官方工具链（Nsight Systems / Nsight Compute）找出这 68% 的算力浪费在哪，并落地优化。

## 工具链：为什么选 Nsight

| 工具 | 定位 | 我用它干什么 |
|---|---|---|
| **nsys** (Nsight Systems) | 系统级时间线，看"时间花在哪" | kernel 时长分类、调用次数、API 开销 |
| **ncu** (Nsight Compute) | kernel 级剖析，看"某个 kernel 为什么慢" | 内存带宽 vs 计算利用率、瓶颈定位 |

> 注意：ncu 需要 GPU 性能计数器权限（宿主机驱动 `RmProfilingAdminOnly=1` 时容器内无权限）。如果遇到 `ERR_NVGPUCTRPERM`，方案：宿主机 root 跑 ncu，或由管理员设置 `NVreg_RmProfilingAdminOnly=0`（需重载驱动）。本文 nsys 数据已足够支撑分析，ncu 作为深入手段演示。

### 采集命令（可复现）

```bash
# 1. 系统级时间线：抓 kernel + CUDA API + 显存事件
nsys profile -o mini_base --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
  --num-steps 40 --warmup-steps 10 --micro-batch-size 4 --amp

# 2. 生成统计报告
nsys stats -r cuda_gpu_kern_sum --format csv mini_base.nsys-rep

# 3. kernel 级剖析（有权限时）
ncu --kernel-name regex:multi_tensor --launch-count 8 \
  --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 --num-steps 5 --warmup-steps 0 --micro-batch-size 4 --amp
```

采集了 40 步（10 warmup + 30 测量），约 30,953 次 kernel 调用。用 nsys 导出的 SQLite 按 kernel 名分类聚合：

```python
# 用 nsys 生成的 sqlite 做 kernel 分类统计（核心代码）
rows = cur.execute("""SELECT K.demangledName, COUNT(*), SUM(K.end-K.start)
                      FROM CUPTI_ACTIVITY_KIND_KERNEL K
                      GROUP BY K.demangledName ORDER BY 3 DESC""").fetchall()
# demangledName 是 StringIds 外键，需 JOIN 查询
```

## 发现 1（核心）：优化器占了近一半的 GPU 时间

40 步训练 kernel 时间分布：

| 类别 | 调用次数 | 总时间 | 占比 |
|---|---|---|---|
| **AdamW 优化器 (multi_tensor)** | 3,200 | 1.000s | **45.2%** |
| matmul (GEMM) | 9,750 | 0.814s | 36.8% |
| dtype copy (autocast 类型转换) | 8,700 | 0.188s | 8.5% |
| softmax (交叉熵) | 100 | 0.081s | 3.7% |
| layernorm | 3,750 | 0.041s | 1.9% |
| 其他 (gelu/fill/embedding…) | — | ~0.09s | ~4% |

**反直觉的事实：125M 小模型上，优化器消耗的 GPU 时间比所有 GEMM 加起来还多。**

### 根因分析

PyTorch 默认 AdamW 的 `step()` 是**逐个算子执行的**，每步大约 9 个 multi-tensor kernel：

1. `param.mul_(1 - lr*wd)` —— weight decay
2. `exp_avg.mul_(β1)` / `exp_avg.add_(grad, α=1-β1)` —— 一阶动量
3. `exp_avg_sq.mul_(β2)` / `exp_avg_sq.addcmul_(grad, grad, ...)` —— 二阶动量
4. 偏差修正（bias correction）的 `mul_` / `div_`
5. `exp_avg_sq.sqrt().add_(eps)` —— denom
6. `param.addcdiv_(exp_avg, denom, value=-lr)` —— 参数更新

**每个 kernel 都要把全部参数全量读写一遍。** 125M 参数 × 3 个状态张量（param + exp_avg + exp_avg_sq）× 4B = 1.5GB，一次读写 = 3GB 流量。

关键点：**优化器是纯内存带宽算子，而 L20 的 GDDR6 带宽只有 864 GB/s**（H100 HBM3 的 1/4）。

理论计算：每步 9 个 kernel × 3GB 流量 = 27GB @ 864GB/s ≈ **31ms/step**——实测每步约 25ms 的优化器时间，与理论吻合（kernel 之间无重叠时）。

L20 的算力（110 TFLOPS）足以让 GEMM 跑在 36.8%，而优化器的内存流量把它拖成了"带宽游戏"。**这是典型的 compute-light / memory-heavy 场景，小模型 + 高带宽比算力比值低的内存型 GPU 最容易踩的坑。**

### 另一个隐藏开销：autocast 的逐 GEMM cast

8.7% 的 `bfloat16_copy_kernel`：BF16 autocast 下，`nn.Linear` 的 **FP32 权重在每次 GEMM 前都要被 cast 成 BF16**。12 层 × 6 个 Linear × 前向+反向 ≈ 150 次/步的小型 copy kernel。这个在后续用 `torch.compile` 或权重预转换可消除。

## 优化：一行代码，fused AdamW

PyTorch 自带 AdamW 的 fused 版本（`fused=True`），把所有更新逻辑合并成**单个 CUDA kernel**，每步从 ~9 次全量扫描变成 1 次：

```python
optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                  weight_decay=cfg.WEIGHT_DECAY, fused=True)
```

### 结果（严格交替复测，同 seed 随机数据）

| 指标 | unfused | fused | 提升 |
|---|---|---|---|
| 吞吐 | 51,700~51,831 tok/s | **60,606~60,617 tok/s** | **+17.1%** |
| MFU | 36.2% | **42.4%** | **+6.2pp** |
| 优化器 kernel 时间 | 1.000s (45.2%) | 0.432s (26.3%) | **-57%** |
| 总 kernel 时间 | 2.210s | 1.643s | **-25.7%** |

> 数据采集方式：unfused / fused 交替各跑 2 轮（每轮 50 测量步 + 10 warmup），取每轮结果。同 seed（42）随机数据，两路 loss 行为一致。
> **口径说明**：优化器"45.2%"是占 GPU kernel 总时间的比例；占 wall-clock（kernel 起止跨度）为 38.5%。
> **为什么不写第一次测的 +33.6%**：首次基线测得 45,371 tok/s，同配置交替复测稳定在 51.7k——首测值偏低（GPU 频率/负载未稳定），提升比例被夸大。以交替复测为准。

优化后的分布：matmul 升到 49.5%（不再是优化器主导），优化器降到 26.3%。

**为什么 fused 省这么多**：合并成一个 kernel 后，参数只读一次、写一次；而且 fused 内核在寄存器/共享内存里完成中间计算，避免了 8 次独立的 DRAM 往返。对 125M 模型，9 次全量扫描 vs 1 次，内存流量直接降到 1/9 量级。

### 正确性验证（固定数据文件，同 seed）

| 步数 | 数据 | Max loss diff | Mean loss diff |
|---|---|---|---|
| 50（仅固定数据） | identity_data.pt | **0.000000** | **0.000000** |
| 200（含随机段） | identity_data.pt | 0.000100 | 0.000010 |

> 使用 `experiments/identity_data.pt`（预生成、加载后不再消耗 RNG），unfused/fused 各跑一遍，逐 step 对比 loss。
> fused kernel 内部归约顺序与 unfused 略有不同，训练早期（loss 尚未饱和）逐位一致；随步数增加浮点噪声累积到 ~1e-4，属正常数值噪声，不影响收敛。面试表述：**"数值等价，非位级一致"**。

### vs Megatron-Core：fused 都打开还快吗？（2026-08-11 公平对比）

之前的对比只测了 Megatron-Core 的默认配置。为了公平，给 `eval/run_megatron_baseline.py` 也加了 `--fused`，两个框架各自 unfused/fused 交替 2 轮（同一天、BF16、TP=1、50 步+10 warmup）：

| 配置 | Round 1 | Round 2 | 稳定值 | mini / Megatron |
|---|---|---|---|---|
| mini unfused | 51,843 | 51,838 | 51,841 | — |
| mini fused | 60,797 | 60,710 | **60,754** | — |
| Megatron-Core unfused | 24,696 | 24,626 | 24,661 | 2.10x |
| Megatron-Core fused | 26,694 | 26,587 | 26,641 | **2.28x** |

结论：
- **mini-megatron 快 2.1x 起步，fused 都打开时快 2.28x**；
- Megatron-Core 开 fused 只 +8%（24.7k→26.6k），因为它的瓶颈不在优化器（Float16Module 包装、层实现开销占主导），fused 治不了它的病；
- 两个脚本用**完全相同的 MFU 公式和吞吐定义**（`B×S×steps/elapsed`），同日交替测量，对比公平；
- 注意：Megatron-Core 绝对吞吐比 2026-07-24 记录（16.4k）高了 ~50%，环境/版本变化所致，跨日期不可比，只能比同日相对值。

### 多卡验证（交替复测，同条件）

| 配置 | unfused | fused | 提升 |
|---|---|---|---|
| TP=2 PP=1 | 26,133~26,708 tok/s | 28,126~29,290 tok/s | **+7.6%** |
| TP=2 PP=2 | 23,186~23,686 tok/s | 23,261~23,764 tok/s | ~0（噪声内） |

> fused 收益随参数切分减少而消失：每 rank 参数变少后，优化器不再是内存带宽瓶颈。

## 剩余瓶颈与下一步

优化后剩余分布：

| 类别 | 占比 | 优化方向 |
|---|---|---|
| matmul | 49.5% | 小 GEMM（B=4 导致 K 维度小）→ 增大 batch、CUDA graphs 隐藏 launch |
| dtype copy | 11.5% | 权重预转换（bf16 副本）、torch.compile 自动融合 |
| AdamW fused | 26.3% | 已经是单 kernel，可尝试 ZeRO-1/多卡分摊 |
| softmax (CE) | 4.9% | 1.38ms/次的 log_softmax，可换 vocab-parallel + 融合 kernel |

另外 L20 单卡的天花板受 GDDR6 带宽限制（864GB/s），理论峰值 MFU 也就是 50% 上下；多卡 PCIe 无 NVLink 是硬伤。真正要追更高 MFU 需要 H100 级别带宽或更大模型。

## 复盘：面试怎么讲这个故事

1. **发现问题**：不是"试了 torch.compile 提速 20%"这种玄学，而是先用 Nsight Systems 量化出 45.2% 的时间在优化器上（数据驱动）
2. **解释根因**：内存带宽模型——优化器是 memory-bound，算了下理论流量和实测吻合
3. **动手优化**：fused AdamW 单 kernel 化，-57% 优化器时间，MFU 36.2% → 42.4%
4. **验证**：固定数据 + 同 seed 的 loss 曲线对比，确认没有改变训练语义

这个故事同时覆盖了三类面试点：**性能分析工具链（nsys/ncu）、硬件理解（内存带宽 vs 算力）、框架内核知识（AdamW 实现、autocast 机制）**——正好对应目标岗位 JD 里"熟悉深度学习框架优化/问题定位相关工具链""熟悉硬件机制"的要求。

## 附录：测试条件与复现（完整）

### 环境

| 项 | 值 |
|---|---|
| 硬件 | 4× NVIDIA L20 48GB（Ada cc8.9，GDDR6 864GB/s，PCIe 无 NVLink） |
| 容器 | NGC PyTorch 26.01（CUDA 13.1，torch 2.10.0a0，NCCL 2.29.2），shm=1G |
| 模型 | 125M GPT（12 层 / 768 hidden / 12 头 / 512 seq），B=4 |
| 精度 | BF16（`torch.autocast`），TF32 开启 |
| 优化器 | AdamW lr=6e-4, wd=0.1, cosine schedule, warmup 10 |

### 复现命令

```bash
# 1. 吞吐对比（随机数据，同 seed=42，交替跑 2 轮取稳定值）
# unfused:
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
  --num-steps 50 --warmup-steps 10 --micro-batch-size 4 --amp
# fused:
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
  --num-steps 50 --warmup-steps 10 --micro-batch-size 4 --amp --fused

# 2. 训练等价性（固定数据文件，同 seed，逐 step 对比 loss）
python3 experiments/synthetic_data.py experiments/synthetic_data.pt
python3 experiments/make_identity.py    # 生成 experiments/identity_data.pt
python3 experiments/compare_convergence.py --compare-fused \
  --data-file experiments/identity_data.pt --steps 50 --warmup 0

# 3. nsys 剖析（-57% 优化器时间的数据来源）
nsys profile -o mini_base --trace=cuda \
  torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
  --num-steps 40 --warmup-steps 10 --micro-batch-size 4 --amp
nsys profile -o mini_fused --trace=cuda \
  torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
  --num-steps 40 --warmup-steps 10 --micro-batch-size 4 --amp --fused
# 然后：nsys stats -r cuda_gpu_kern_sum，或导出 sqlite 按 kernel 名聚合
```

### 已知的口径与波动

- 提升比例随**总吞吐水平**波动：同 seed 随机数据 51.7k→60.6k（+17.1%）；固定数据文件任务 32.7k→36.1k（+10.5%），绝对收益 ~9k tok/s 相近
- 多卡测试同样需要交替复测（PCIe 通信下吞吐波动更大，TP=2 PP=2 的 ±2% 为噪声）
- 数值上 fused 非位级一致：kernel 归约顺序不同，训练后期 loss 差 ~1e-4，属正常噪声
