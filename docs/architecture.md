# 架构

> mini-megatron 的系统架构、模块划分、设计决策。

---

## 总体架构

mini-megatron 是 Megatron-LM 核心并行策略的教学实现；wired training path 约 830 行，覆盖：
- **Tensor Parallelism (TP)**：ColumnParallel + RowParallel 切分
- **Pipeline Parallelism (PP)**：默认 non-interleaved 1F1B（serial 仅作 legacy 对照）
- **Data Parallelism (DP)**：all-reduce 梯度同步
- **BF16 Mixed Precision (AMP)**：`torch.autocast`
- **ZeRO-1 Optimizer**（参考实现，未接入）
- **Sequence Parallelism**（参考实现，未接入）

---

## 模块划分

```
mini-megatron/
├── main.py                  # 入口 + 训练循环（TP/PP/DP/AMP 路径）
├── config.py                # 模型 + 训练超参
├── checkpoint.py            # Save/load（参考）
├── model/                   # 模型组件
│   ├── embedding.py         # 词嵌入 + 位置嵌入
│   ├── transformer.py       # Attention, DecoderLayer, Decoder, GPT
│   └── loss.py              # Cross-entropy
├── parallel/                # 并行原语（wired）
│   ├── tensor_parallel.py   # ColumnParallelLinear, RowParallelLinear
│   ├── pipeline_parallel.py # non-interleaved 1F1B + legacy serial
│   ├── data_parallel.py     # all-reduce 梯度
│   ├── process_groups.py    # TP/PP/DP 通信组
│   └── distributed_optimizer.py  # ZeRO-1（参考）
├── comm/                    # 通信原语
│   ├── all_reduce.py        # autograd Function
│   ├── send_recv.py         # P2P（参考）
│   ├── sequence_parallel.py # SP（参考）
│   └── overlap_*.py         # 通信-计算 overlap（参考）
├── eval/                    # 对比基线
│   └── run_megatron_baseline.py  # Megatron-Core baseline
├── experiments/             # artifact、数值等价、配对 benchmark、bundle 校验
├── tests/                   # 38 项实现与实验资产测试
├── results/                 # aggregate 与带 checksum 的 clean evidence bundle
└── README.md
```

---

## 数据流（训练一步）

### 非 PP 路径（TP/DP）

```
main.py (TP/DP)
  └→ make_data_iterator() → tokens, labels
  └→ model(tokens, labels) → logits, loss
       └→ Embedding → Decoder → LayerNorm + LMHead
            └→ ColumnParallelLinear / RowParallelLinear
  └→ F.cross_entropy(logits, labels) → loss
  └→ loss.backward()
  └→ allreduce_grads(model, dp_group)  # DP 同步
  └→ optimizer.step()
```

### PP 路径

```
main.py (PP)
  └→ train_pipeline_1f1b() in parallel/pipeline_parallel.py（默认，--schedule serial 用旧版）
       ├→ build_1f1b_schedule(): 每 stage 的 op 序列（warmup F → F/B 交替 → drain B）
       ├→ Stage 0: forward embedding + decoder_layers[0..N/PP] → send to stage 1
       └→ Stage 1: recv → forward → send
       └→ Last stage: recv → forward + CE loss → backward → send grad
       └→ Stage 0: recv grad → backward → optimizer.step
       └→ look-ahead: recv 提前一个 op 发出，与上一个 op 的计算重叠
```

---

## 关键设计决策

### 1. 不依赖 HuggingFace
纯 PyTorch。无 transformers / accelerate。
**原因**：HF 抽象层隐藏了核心并行机制。教学目的要让用户看清每一行代码。

### 2. F.scaled_dot_product_attention
attention 用 PyTorch 内置的 SDPA。
**原因**：自动选择最优 kernel（Flash Attention 等），避免手写 attention。

### 3. 1F1B 流水线（默认）+ serial 参考
PP 默认用 Megatron 的 1F1B 调度：每 stage 先做 `pp-rank-1` 个 warmup forward，
然后交替 forward/backward，最后 drain。bubble 从 serial 的 `(pp-1)/pp` 降到
`(pp-1)/(2m+pp-1)`（pp=2, m=50 时 ~1%）。
**原因**：1F1B 让每个 stage 全程忙碌（backward 填充 forward 的 bubble），
同时支持任意 pp >= 2；旧的 serial 锁步调度只实现了 pp=2（中间 stage 无代码）。
`--schedule serial` 保留作为对比基线。

PP stage 之间不共享 PyTorch autograd 图：前一 stage 发送 detached activation，下一
stage 以 `requires_grad=True` 将它重建为图根；最后 stage 计算 loss 后把 activation
gradient P2P 传回，前一 stage 执行 `x.backward(received_grad)`。这条链路是 PP 真正
“跑通反向”的关键，而不是只把 forward activation 发到下一张卡。

### 4. 标准 init（std=0.02）
所有 Linear 用 `N(0, 0.02)`，不用 Megatron 的 scaled init。
**原因**：scaled init 针对 1B+ 大模型优化，125M 反而学得慢。

### 5. 无 bias
所有 Linear 设 `bias=False`。
**原因**：少参数，简化训练。

### 6. Tied embeddings
embedding 和 LM head 共享权重。
**原因**：少 38M 参数（125M 模型中），效果不变。

---

## TP 切分策略

```
Q/K/V 投影: ColumnParallel（输出维度切分）
  ↓
Attention 计算（每个 GPU 一组 head）
  ↓
输出投影: RowParallel（输入维度切分 + all-reduce）
  ↓
FFN up: ColumnParallel
FFN down: RowParallel
```

参考 Megatron-LM 论文。

---

## PP 切分

按层平均分到 N 个 stage：
```
PP=2, 12 层 → stage 0 = layers[0..5], stage 1 = layers[6..11]
```

warmup：stage 0 先做 N-1 个 forward，再开始 normal pipeline。

`build_1f1b_schedule()` 对每个 stage 生成 warmup forward、F/B 交替和 drain。默认
`1f1b` 支持任意 PP >= 2；`serial` 只保留为 PP=2 的旧对照路径。

---

## 通信原语

| 操作 | 实现 | 文件 |
|------|------|------|
| all-reduce（DP） | `dist.all_reduce` | parallel/data_parallel.py |
| all-reduce inside RowParallel | `autograd.Function` | comm/all_reduce.py |
| send/recv（PP） | `dist.isend/irecv`（异步 + look-ahead） | parallel/pipeline_parallel.py |
| broadcast | `dist.broadcast` | parallel/process_groups.py |

---

## 训练循环关键参数

| 参数 | 值 | 文件 |
|------|------|------|
| LR | 6e-4 | config.py:LEARNING_RATE |
| warmup | linear | main.py:get_lr_lambda |
| decay | cosine | main.py:get_lr_lambda |
| AdamW β | (0.9, 0.999) | 默认 PyTorch |
| weight_decay | 0.1 | config.py:WEIGHT_DECAY |
| max_grad_norm | 1.0 | 未启用（参考代码里有，main.py 未调） |
| dropout | 0 | config.py（已禁用） |

---

## 如何验证这些设计

- `tests/test_pipeline_1f1b.py`：PP=2 和 PP=4 的 CPU/Gloo 多进程训练，更新后的每个
  参数必须与单进程参考一致；
- `tests/test_experiment_artifacts.py`：bundle checksum、QKV layout round-trip、
  idle-GPU 前置检查和配对统计规则；
- clean L20 环境中完整 suite 为 **38 passed / 11.28s**。

性能层面还有独立的共享权重、固定输入、数值等价与五组交替配对流程；它不与单元测试
混为一谈。详见 [benchmark guide](benchmarks.md) 和
[experiment protocol](experiment-protocol.md)。
