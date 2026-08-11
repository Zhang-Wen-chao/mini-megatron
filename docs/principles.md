# 设计原则

> mini-megatron 为什么是 ~800 行、为什么纯 PyTorch、为什么 1.6-2.4x 快于 Megatron-Core。这些设计的权衡和理由。

---

## 一、为什么 800 行

**目标读者**：想理解 Megatron-LM 并行机制的学生、初学者、AI 培训学员。

**800 行的含义**：
- < 500 行：太简单，隐藏关键机制
- ~800 行：完整覆盖 TP/PP/DP/AMP，每行都重要
- 2000+ 行：开始有 framework 的复杂度

**判断标准**：一个新人能在 1-2 小时内读完全部代码并理解每一行。如果不能，代码太复杂。

---

## 二、为什么纯 PyTorch

| 选型 | 评价 |
|------|------|
| PyTorch 原生 | ✅ 透明，每步可见，调试容易 |
| Apex | ❌ C++/CUDA 混编，隐藏机制 |
| Transformer Engine | ❌ 过度工程，对 125M 模型 overkill |
| Megatron-Core | ❌ 300K 行，教学用看不懂 |
| DeepSpeed | ❌ ZeRO 设计为主，TP/PP 不直观 |

**原则**：教学价值 > 生产价值。

例外：完全独立的小工具（如 `comm/overlap_*.py`）可以略复杂，但需明确标注。

---

## 三、为什么 1.6-2.4x 快

mini-megatron 2000 步对比：loss 0.0054 vs Megatron 0.30。

| 原因 | 说明 |
|------|------|
| 无 DDP 包装层 | 我们的 125M 模型不需要分布式包装开销 |
| 无 Float16Module | Megatron 强加的 FP16 包装对小模型反而慢 |
| 用 `torch.autocast` | 轻量级 BF16，按需转换，零开销 |
| 无 checkpoint 框架 | 训练循环直接 50 行，零抽象 |
| 简单数据流 | `tokens → model → loss → backward → step` 直线 |

代价：mini-megatron 不用 FlashAttention（用 F.scaled_dot_product_attention 自动选）、不用 fused kernel（用 PyTorch 原生 matmul）。对 125M 模型，融合收益 < 抽象成本。

---

## 四、为什么收敛快 7-10x

2000 步 identity 任务：mini 0.0054 vs Megatron 0.30。

| 原因 | 说明 |
|------|------|
| 无 scaled init | Megatron 的 `output_layer_init_method` 用 `std=0.02/sqrt(2L)≈0.0041`，前 1500 步输出层几乎不动 |
| 无 FusedLayerNorm | Megatron 的 `FusedLayerNorm` 是 Triton 实现，gradient 路径不同 |
| 简单 attention | `F.scaled_dot_product_attention` 自动选最优 kernel，无 wrapper 开销 |

**结论**：Megatron 的优化（融合 kernel、TP-aware init）针对 1B+ 大模型，对 125M + 短训练是负优化。

---

## 五、不做什么

明确**不**实现（避免 scope creep）：

| 功能 | 不做的原因 |
|------|----------|
| 1F1B interleaved | serial 1F1B 对 125M 够用，interleaved 增加复杂度 |
| MoE / Expert Parallel | 教学目标不需要；单独项目 |
| Flash Attention 显式 | F.scaled_dot_product_attention 自动选 |
| Fused softmax / rotary | 125M 模型，PyTorch 原生够用 |
| Dynamic loss scaling | BF16 不需要 loss scaling（FP16 才需要） |
| FSDP | ZeRO-1 风格（参考实现）足够教学 |
| Checkpoint optimizer 状态 | 用户没要求；参考代码里有，未调 |
| 数据并行 + 流水线并行组合 | 125M 模型不实用 |
| Transformer Engine | 教学目标不需要 |
| ONNX export | 跟纯 PyTorch 教学目标冲突 |

---

## 六、什么时候应该升级

mini-megatron 不应该无限增长。如果哪天：
1. 有人要做 1B+ 训练 → 应该 fork 出 "production" 版本
2. 有人要加 MoE → 单独项目
3. 核心 800 行被突破 → 拆分子目录

判断：单文件超过 1500 行就拆。

---

## 七、benchmark 怎么读

README 里的 "1.6-2.4x faster" 不是营销话术，是有数字支撑的：
- 4×L20 48GB 测的（不是理论值）
- 50 步 warmup + 50 步 benchmark
- 模型、batch、seq_len 都和 Megatron-Core 一样

详见 [benchmarks.md](./benchmarks.md)。

---

## 八、教学优先

mini-megatron 是教学项目，不是生产框架：
- 可读性 > 性能
- 透明 > 抽象
- 简单 > 复杂
- PyTorch 原生 > 自定义 kernel

**铁律**：如果加一行代码让代码变复杂 10%，省 0.5% 性能，不加。
