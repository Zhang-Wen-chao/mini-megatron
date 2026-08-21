# 2026-08-17/18 L20 实验证据 ledger

这是 L20 实验会话的可审计记录，不替代[实验协议](experiment-protocol.md)。原始 2026-08-17
样本仍有历史佐证价值，但其 manifest 记录 `source_tree_clean=false`，因此只是临时证据。
下面的当前结论来自 2026-08-18 clean-tree rerun，并取代这些样本作为正式依据。

## 环境与已完成检查

- 宿主机/容器：L20 host，zhangwenchao-megatron（nerdctl），4 × NVIDIA L20；
  NGC PyTorch 26.01、CUDA 13.1、PyTorch 2.10、Megatron-Core 0.18。
- 历史 profile 使用 BF16；当前公平比较使用 125M GPT、TP=1/PP=1、FP32、micro-batch 4、
  sequence 512、30 warm-up 和 200 measured steps。
- Clean rerun 源码 commit：`ad82d7edf9a1a11f61672aae492697cf15434b85`
  （对应本地源码 commit `f402fea` 的 L20-history 版本）。
- 测试套件：容器内运行 `python3 -m pytest -q`，在 clean rerun 前完成
  **38 passed in 11.28s**。这是完整 suite，不是仅 macOS 可运行的 partial check。
- 下列所有样本均通过 runner 的 idle-GPU preflight。一个更早的 active-GPU 样本保留在
  独立 bundle 中，但已从正式汇总排除。

## 配对吞吐证据

每种对比都记录了 5 组交替配对。全部数值单位为 tokens/s；sd 为 sample standard deviation。
完整 aggregate 保存在 L20 的 results/aggregates/。

| 对比 / 条件 | 左侧结果 | 右侧结果 | 配对比 | 它能够支持什么 |
|---|---:|---:|---:|---|
| mini fused vs Megatron-Core custom loop，125m-s512-b4-bf16-fused-200x30 | mini mean 60,670，median 60,702，sd 75.58 | baseline mean 26,503，median 26,541，sd 130.18 | mini/baseline mean **2.289x**，范围 2.272-2.305 | 仅是诊断：两条非等价脚本的配对测量，不是框架性能结论。 |
| mini fused vs mini unfused，mini-125m-s512-b4-bf16-200x30 | fused mean 60,675.8，median 60,674，sd 34.07 | unfused mean 51,823.4，median 51,829，sd 67.07 | fused/unfused mean **1.1708x**（+17.08%），范围 1.1692-1.1727 | fused optimizer path 带来的可重复 mini 内部局部吞吐增益。 |

第一种对比并非严格的同权重/同语义模型等价，已经明确撤回为性能主张。它使用了独立初始化、
独立生成的随机 identity-token batch 和不同的原生 forward graph；所以它测到的是两条脚本的
总 step 成本，并不是 apples-to-apples 的框架差异。它**不能**证明 mini-megatron 普遍快于
Megatron，也不覆盖更大模型、多机任务、生产功能或训练质量。fused/unfused 的随机数据 loss
在最后一步比较接近，但这不能替代固定输入的等价协议。

## 公平 TP=1 FP32 对比（2026-08-18，clean tree）

这次实验以刻意共享的模型合同，替代上面无效的跨框架性能主张：

- 12 layers、hidden 768、12 heads、FFN 3072、learned absolute positions、pre-LayerNorm、
  GELU、无 dropout、无 bias 的 QKV/projection/MLP linear，以及 causal next-token
  cross entropy。
- mini 与 Megatron-Core 都包含 162,633,216 参数。映射全部 101 个 parameter tensor；
  QKV 在 mini 的 all-Q then all-K then all-V 布局和 Megatron-Core 的按 head
  Q_i then K_i then V_i 布局之间显式转换。
- 不可变 L20 artifact 目录：
  /mnt/storage01/zhangwenchao02/repos/mini-megatron-test/artifacts/fair-tp1-20260818-v2/。
  其中包含由相同权重构造的各框架 checkpoint，以及 230 个固定 next-token batch。SHA-256：
  mini checkpoint
  93d634e9e45699266633946a5f9436f08369f160360fc6c229a71c8fdad47619；
  MCore checkpoint
  2c7609572698f0cbb660b62a3ddb51fb49e3730b27b74a626ac532c61b250222；
  batch artifact 3e0f943f435db4bac634f57af2b6a6a21f0f6ecbe2c0f15d5c7efecaa49cabc5。
- clean-tree FP32 equivalence gate 已通过：初始映射权重精确为 0 差异；logits relative
  L2 为 3.5498e-4，worst gradient relative L2 为 3.5508e-4，一次 optimizer step 后
  parameter relative L2 为 6.3139e-5，均在预先声明的 5e-4、5e-4、1e-4 阈值内。

在空闲 GPU 上以这些 artifact、standard unfused AdamW、FP32、30 warm-up 与 200 measured
steps 记录 5 组 ABBA/BAAB-style pair。每个 bundle 都通过 checksum validation，且记录
`source_tree_clean=true`；aggregate 如下：

| 指标 | mini | Megatron-Core | mini / MCore |
|---|---:|---:|---:|
| Mean throughput | 32,669.6 tok/s | 27,704.8 tok/s | **1.179204x** |
| Median throughput | 32,664 tok/s | 27,716 tok/s | **1.178825x** |
| Sample standard deviation | 17.34 | 24.39 | 0.001047 |
| Range | 32,654-32,690 | 27,672-27,731 | 1.178079-1.180554 |
| Peak allocated memory | 4.52 GB | 5.13 GB | — |

不可变 aggregate 为
`results/aggregates/fair-tp1-fp32-unfused-clean-ad82d7e.json`（SHA-256
`132cda05cc47c0042759eb053380df31b08efbd7b704989e47e8881ef731a213`）。其 10 个 source
bundle 位于 `results/runs-clean-ad82d7e/`，pairs 01–05。这只支持一个刻意收窄的表述：
**在这台 L20、共享的无 bias 125M GPT 合同、TP=1/PP=1、FP32 与 standard AdamW 下，
mini 的吞吐是 matching Megatron-Core path 的 1.179204x。**它不支持默认 MCore、BF16、
fused optimizer、更大模型、TP/PP/多 GPU scaling、生产 workload 或训练质量的主张。

此前 dirty-tree aggregate 保留为历史复现记录，但不与当前结果合并。旧的 2.289x 跨框架
BF16 测量仍是非等价诊断，而非性能主张。

BF16 被明确排除在公平性能结论外。在相同初始权重与一个 batch 下，logits relative L2 为
5.3204e-3，worst gradient relative L2 为 7.0706e-2，未通过声明的 FP32 parity gate。
在解决这一数值差异之前，两种实现仍可作为相同初始条件下的 BF16 性能观察进行比较，但不能
标注为语义等价训练。

## Nsight Systems 原始证据

两份 clean-tree profile bundle 已版本化在本仓库 `results/runs-clean-ad82d7e/` 下；
`git pull` 可获得原始 `.nsys-rep`、SQLite、CSV exports、manifest、commands、checksums
和保守 analyzer output。L20 实验归档中也保留了一份相同副本：
`/mnt/storage01/zhangwenchao02/repos/mini-megatron-test/results/`。profile 时间不纳入
吞吐统计。

clean-tree profile 使用与吞吐实验相同的 checkpoint/batch 和 FP32 合同，包含 10 warm-up
与 20 measured steps：

| Bundle | rank0.nsys-rep | rank0.sqlite | kernel time | 保守 kernel-time 分类 | SHA-256 (.nsys-rep / .sqlite) |
|---|---:|---:|---:|---|---|
| `20260818T003611Z-fair-tp1-fp32-clean-profile-mini` | 3.8 MB | 12 MB | 1.8362 s | GEMM 61.54%，copy/cast 4.09%，unclassified 34.36% | `52ff06a0d9099f5b879a3c1fab72b96027fece3e2293e1c994f4b6c37296c902` / `cde5afc1de50c83d48dbddb49d6e951ad364ddd615fbb5b697d9bfea45279ec0` |
| `20260818T003641Z-fair-tp1-fp32-clean-profile-mcore` | 4.2 MB | 13 MB | 2.0363 s | GEMM 50.18%，copy/cast 10.51%，unclassified 39.31% | `48780b2bb1579d316598f834620951dfbe71af6d986d0e470249905961b59520` / `2f37c97a9748072965f55f5cf2d2d2a30d20a0d597da84c2c5c94afdb26a94fe` |

这些是 kernel-time 描述，而不是吞吐差异的因果证明：分类器刻意将未匹配 kernel 保持为
unclassified，也不会从通用 elementwise 名称推出 unfused-AdamW 归因。

历史 BF16 profile 单独保留：

| Bundle | rank0.nsys-rep | rank0.sqlite | SHA-256 (.nsys-rep / .sqlite) |
|---|---:|---:|---|
| 20260817T153432Z-mini-125m-tp1-bf16-unfused-profile-p01 | 5.3 MB | 15 MB | 6353186078149f68ba871d50b15f8b5ca0c1e427a26083ea222ff4fa4eae2fc1 / 535ac642f093e66c440f621200bbd6a983c8db142f592d04f6f8d26a95e89b25 |
| 20260817T153506Z-mini-125m-tp1-bf16-fused-profile-p01 | 2.9 MB | 8.1 MB | c131f1fda66b3b8308f41b9437ea02154631b576746962f9f5140457d5feead5 / c32591aa2a4fe82f107fe548e64f3c043c65ffe5b7f997e7b533788bc222092b |

保守 CSV analyzer 对 unfused capture 报告 1.9083 s total GPU-kernel time（GEMM 42.53%，
copy/cast 10.28%，unclassified 47.19%），且不会把通用 elementwise kernel 归为 AdamW。
fused capture 中可识别显式 fused AdamW kernel：432.31 ms、450 invocations、占 1.6428 s
total kernel time 的 26.32%（GEMM 49.52%，copy/cast 12.00%，unclassified 12.16%）。
这些都是 kernel-time share，不是 wall-clock share。

## 扩大结论前还需要什么

1. 在作 scaling claim 前，将 shared-weight conversion 与 numerical parity gate 扩展到 TP/PP。
2. 解决 BF16 parity failure，才能将 BF16 跨框架结果标为语义等价。
3. 若需要这些范围的结论，则对更大模型和 production-relevant configuration 重复协议。
4. 在不可变 archive 或 Git LFS 中保存 raw report，并在本 ledger 中持续记录 checksum。
