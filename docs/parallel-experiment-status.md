# 125M 并行实验总览：计划、结果、边界与资产

> 先看这篇，再引用性能数字。它统一说明 125M mini-megatron 与 matching Megatron-Core custom-loop 的 TP/PP 矩阵：测了什么、得到了什么、哪些结论可信、哪些不能说。方法规则见实验协议，单卡与 Nsight 细节见 Benchmark。

## 一句话结论

- **唯一无条件的性能结论：**在 clean-tree 的 L20、125M、FP32、standard unfused AdamW、共享权重和固定输入下，TP=1、PP=1 的 mini 为 **32,669.6 tok/s**，matching MCore custom-loop 为 **27,704.8 tok/s**，mini/MCore 为 **1.179204x**，约快 17.9%。
- 125M 的四格 TP/PP 矩阵已经跑完。TP=2、PP=1 为 **0.986352x**，mini 约慢 1.36%；TP=1、PP=2 为 **1.201991x**；TP=2、PP=2 为 **0.974974x**，mini 约慢 2.5%。PP 两格采用后验、探索性、三窗口复现的校准，因此只能称为**条件性 matching-custom-loop 观察**。

## 1. 比较合同

所有多卡格共享 125M GPT：12 layers、hidden 768、12 heads、FFN 3072、无 bias、learned absolute position、pre-LN、GELU、无 dropout；FP32、standard unfused AdamW、共享 canonical 权重、固定 next-token batch。每个样本为 30 warm-up + 200 measured update，每 update 8 micro-batch（16,384 token），mini/MCore 以 ABBA 顺序交替五对。

PP 两边比较的是明确标记的 **matching custom-loop PP comparison**：显式 P2P、非交错 1F1B、每 stage 一次 FP32 unfused AdamW。它不等于默认或完整 Megatron-Core production training stack。

每个多卡拓扑均保存固定 artifact、数值报告、五对吞吐汇总、独立 multi-rank Nsight profile；每条运行保存命令、环境、日志、metrics 与 SHA-256 checksum。Nsight profile 只作事件结构资产，绝不替代吞吐测量。

## 2. 四格状态

| 拓扑 | GPU | 数值状态 | 5 对吞吐：mini / MCore | profile | 正确表述 |
| --- | ---: | --- | --- | --- | --- |
| TP=1、PP=1、DP=1 | 1 | 原始 gate 通过，clean tree | 32,669.6 / 27,704.8 tok/s；**1.179204x** | 已保存 | 精确单卡合同内，mini 快约 17.9%。 |
| TP=2、PP=1、DP=1 | 2 | 一步参数 relative-L2 略超原门槛；吞吐前声明 1.25e-4 校准 | 12,220.8 / 12,390.2 tok/s；**0.986352x** | 已保存 | 该 TP 合同内 mini 慢约 1.36%，不是通用排名。 |
| TP=1、PP=2、DP=1 | 2 | 原逐 tensor gate 未过；后验三窗口 calibration 复现 | 54,402.8 / 45,260.6 tok/s；**1.201991x** | 每实现 2 个 rank report | 条件性观察：此 fixed-artifact custom-loop 内 mini 约快 20.2%。 |
| TP=2、PP=2、DP=1 | 4 | 原逐 tensor gate 未过；后验三窗口 calibration 复现 | 16,067.8 / 16,480.8 tok/s；**0.974974x** | 每实现 4 个 rank report | 条件性观察：此 fixed-artifact custom-loop 内 mini 约慢 2.5%。 |

PP 的 calibration 含义必须完整保留：在 offset 0 观察到逐 tensor relative-L2 对接近零的 LayerNorm bias 不稳定后，才选择 global metric / threshold；随后独立在 offset 8、16 复现。它不是把原始失败改写为原始 gate 通过。

## 3. 能说与不能说的结论

**能说：**mini 已实现并测试 TP、非交错 1F1B PP、DP process group 和 AMP。125M 四种 TP/PP 拓扑均已有固定 artifact、五对交替吞吐与每个 CUDA rank 独立 Nsight 原始资产。单卡 clean FP32 合同有可明确引用的 17.9% 优势；多卡方向并不一致。

**不能说：**“mini-megatron 普遍比 Megatron-Core 快”、“完整 MCore production stack 更慢”、“PP 原始数值门槛已通过”、BF16/大模型/多机/训练质量结论，或从 Nsight trace 的 absolute time 推导吞吐因果。

source_tree_clean=false 只表示运行时源码有未提交改动，不是 GPU、输入或结果污染。多卡 bundle 的实际运行提交记录为 af6fb22c…；TP=2、PP=1 的既有关键源码也逐文件 hash 对齐至 b624721。这使资产可审计，但不替代 clean-tree 的独立复跑。

## 4. 真实问题与处理

| 问题 | 处理 | 对结论的影响 |
| --- | --- | --- |
| GPU 1/3 跨 NUMA 异常 | 不混入结果；保留诊断、命令、环境，改用稳定拓扑。 | 避免把系统异常作为框架性能。 |
| 一条早期 TP=2 mini 异常长运行 | 中止后保留诊断；另跑完整五对。 | 异常条目未进入统计。 |
| PP 原逐 tensor gate 失败 | 原始失败保留；校准单独标记为后验、探索性、三窗口复现。 | PP 结果只能条件性表述。 |
| Nsight 会扰动执行 | profile 独立保存，不用 absolute time 替换吞吐。 | 只能解释结构，不能给出速度因果。 |
| profile launcher 不可靠 | 每个 CUDA rank 使用独立 nsys wrapper。 | 避免只采到 torchrun 而漏掉 CUDA workload。 |

## 5. 资产在哪里，怎样复查

单卡 clean anchor 已在本仓库：

~~~text
results/runs-clean-ad82d7e/
results/aggregates/fair-tp1-fp32-unfused-clean-ad82d7e.json
~~~

多卡 archive 在 L20 容器仓库：

~~~text
/mnt/storage01/zhangwenchao02/repos/mini-megatron-test/
  evidence/fair-125m-parallel-20260821/
    artifacts/     # 权重、shard、固定 batch、manifest
    parity/        # 原始 gate 与三窗口 calibration
    benchmarks/    # 每格 10 个 immutable run bundle
    profiles/      # 每个 CUDA rank 的 .nsys-rep/.sqlite/CSV
    reports/       # five-pair summary、条件性结论、profile index
    ledger/        # 只追加的路径、大小、SHA-256
  diagnostics/     # smoke、异常、Nsight 与失败记录
~~~

四卡关键文件：

~~~text
reports/tp2-pp2-calibrated-five-pair-summary.json
reports/tp2-pp2-dp1-conditional-conclusion.json
reports/tp2-pp2-dp1-multirank-profile-index.json
profiles/20260821T1309Z-tp2-pp2-mini-multirank-nsys/
profiles/20260821T1310Z-tp2-pp2-mcore-multirank-nsys/
~~~

四卡 summary、conclusion、profile index 的 SHA-256 分别为 932793451a64…、4c19fac053c5…、cfe7c10d1a10…。用 Nsight GUI 前先复制 .nsys-rep 到仓库外，避免改写归档元数据；以 profile index 的 SHA-256 校验副本。

## 6. 后续真正值得做什么

125M 的既定 TP/PP 矩阵已经完整，没有必须补齐的格子。扩大可信范围的后续工作是：

1. clean-tree 独立复跑多卡矩阵，增加重复性；
2. 定位 PP 原逐 tensor parity 失败根因，再以预注册 gate 复验；
3. 在相同协议下扩展 BF16、较大模型、真实固定 train/validation split 和 held-out PPL；
4. 研究 TP=2 小差异时，用低扰动 CUDA Event 分段，不把 Nsight absolute time 当答案。

## 7. 文件怎么读

| 想知道什么 | 看哪里 |
| --- | --- |
| 计划、四格完成度、结论、问题、下一步 | 本文 docs/parallel-experiment-status.md |
| 单卡主结论、TP=2 诊断与复现规则 | docs/benchmarks.md |
| 证据、bundle、ledger 的发布规则 | docs/experiment-protocol.md |
| 单卡原始 evidence 与 checksum | docs/experiment-results-2026-08-17.md、results/runs-clean-ad82d7e/ |
| 多卡 artifact、bundle、Nsight、ledger | L20 evidence/fair-125m-parallel-20260821/ |
| 实现、runner、profile 与汇总脚本 | experiments/ |
