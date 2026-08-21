# Benchmark 与复现

> 本文把当前可审计结论与探索性历史记录分开。吞吐数字只有在模型、数据、数值等价、
> 重复次数和原始证据都被记录时，才值得被引用。

## 当前受控结论（2026-08-18，clean tree）

正式跨框架实验是一次刻意收窄范围的受控实验：在空闲 L20 主机上，将 mini-megatron
与 matching Megatron-Core custom-loop path 比较；它不是对默认生产 Megatron-Core 的比较。

| 控制项 | 固定内容 |
| --- | --- |
| 源码 | clean L20 source commit ad82d7e |
| 并行与精度 | 1 张 L20，TP=1，PP=1，FP32 |
| 模型 | 125M 无 bias GPT：12 layers、H=768、12 heads、FFN=3072、learned positions、pre-LN、GELU、无 dropout |
| 初始状态 | 两边各 162,633,216 参数；101 个 tensor 从共享权重映射；QKV 显式布局转换 |
| 输入 | 相同的不可变 synthetic next-token artifact：230 batches，B=4，S=512 |
| Optimizer | standard unfused AdamW |
| 计时 | 30 warm-up + 200 measured steps，5 组 idle-GPU 交替配对 |

先通过数值等价门槛，再开始计时：

| 比较项 | 观测值 | 预先声明的阈值 |
| --- | ---: | ---: |
| 初始映射权重 max diff | 0 | exact match |
| Logits relative L2 | 3.5498e-4 | 5e-4 |
| Worst gradient relative L2 | 3.5508e-4 | 5e-4 |
| One-step parameter relative L2 | 6.3139e-5 | 1e-4 |

| 指标 | mini-megatron | matching MCore path |
| --- | ---: | ---: |
| 平均吞吐 | 32,669.6 tok/s | 27,704.8 tok/s |
| 中位吞吐 | 32,664 tok/s | 27,716 tok/s |
| 样本标准差 | 17.34 | 24.39 |
| 配对吞吐比 | **1.179204x** | 范围 1.178079–1.180554 |
| Peak allocated memory | 4.52 GB | 5.13 GB |

**能够支持的结论：**在这一精确的共享模型、共享输入、FP32 合同下，mini 的匹配路径
吞吐是 matching MCore custom-loop path 的 1.179204x。

**不能支持的结论：**通用框架排名、默认 MCore、BF16、fused optimizer、TP/PP/多卡扩展、
大模型或多机训练、生产负载行为，或训练质量。

BF16 被有意排除：在同一初始条件下，logits relative L2 为 5.3204e-3，worst gradient
relative L2 为 7.0706e-2，未通过 FP32 使用的 parity gate。

## 如何检查原始证据

聚合结果位于：

~~~text
results/aggregates/fair-tp1-fp32-unfused-clean-ad82d7e.json
~~~

5 组配对 bundle 与 2 组 profile bundle 位于：

~~~text
results/runs-clean-ad82d7e/
├── ...-fair-tp1-fp32-clean-mini-*/
├── ...-fair-tp1-fp32-clean-mcore-*/
├── 20260818T003611Z-fair-tp1-fp32-clean-profile-mini/
└── 20260818T003641Z-fair-tp1-fp32-clean-profile-mcore/
~~~

每个 bundle 均包含 manifest、精确命令、环境、stdout/stderr、metrics、SHA-256 checksum；
profile bundle 另有原始 .nsys-rep、SQLite、CSV 导出和保守分析 JSON。引用前必须校验：

~~~bash
python3 experiments/validate_run_bundle.py \
  results/runs-clean-ad82d7e/<run-id>
~~~

请把原始 .nsys-rep 复制到仓库外再用 GUI 打开。profiling 软件可能更新报告元数据；
Git 版本及其 checksum 才是不可变证据。

[证据 ledger](experiment-results-2026-08-17.md) 记录了软件版本、checksum、每一组复现
统计和原始 profile 文件名。

## Nsight Systems 能补充什么，不能补充什么

profile 复用了相同的共享 FP32 checkpoint 与 batch，但它是单独执行的 10 warm-up + 20
measured-step capture。它用于描述工作分布，不是吞吐样本。

| Kernel-time 描述 | mini | MCore |
| --- | ---: | ---: |
| GPU kernel 总时间 | 1.8362 s | 2.0363 s |
| GEMM | 61.54% | 50.18% |
| Copy/cast | 4.09% | 10.51% |
| 未分类 | 34.36% | 39.31% |
| 稳定区间 GPU gap | 49.981 ms (2.717%) | 139.966 ms (6.255%) |

通用的 vectorized_elementwise_kernel 既可能是 GELU、residual、LayerNorm，也可能是
optimizer step。分析器将这部分保留为未分类，而不把它强行标成 AdamW。因此 profile
只能作为与吞吐方向一致的描述性证据，不能证明某个 kernel 就是唯一因果瓶颈。

## TP=2 预备实验与 Nsight 诊断（2026-08-21，dirty tree，不是发布结论）

这是一组为后续多卡 clean campaign 准备的 TP=2 预备实验，不改变上面的单卡 clean
主结论。它使用同一份 125M shared-contract artifact：FP32、`TP=2, PP=1, DP=1`、
两张空闲 L20（GPU 0/1）、`B=4`、`S=512`、每个 update 8 个 micro-batch（16,384
tokens）、30 次 warm-up、200 次 measured update，以及 standard unfused AdamW。

数值等价门槛已先通过；五组 mini/MCore 交替配对的吞吐如下：

| 指标 | mini-megatron | matching MCore path | mini / MCore |
| --- | ---: | ---: | ---: |
| 平均吞吐 | 12,220.8 tok/s | 12,390.2 tok/s | **0.986352x** |
| 相对差异 | - | - | mini 慢约 **1.36%** |
| 单对范围 | 12,104–12,319 tok/s | 12,351–12,453 tok/s | 0.971975–0.997409x |

这五组配对都显示 mini 未超过 MCore，因此此合同下不能声称 TP=2 更快。每个 measured
update 的平均差约为 18 ms。源树在采样时为 dirty（`source_tree_clean=false`），因此它
只能作为**预备统计摘要**；在 runner/workload 提交并清理工作树后，必须按同一合同重跑
五组配对，才可以升级为发布结论。远端可审计汇总位于：

~~~text
evidence/fair-125m-parallel-20260821/reports/
  tp2-pp1-gpu01-preliminary-five-pair-summary.json
~~~

为解释这 1.36% 的差异，另行执行了两份短 Nsight Systems capture（2 warm-up + 12
measured update）。它们**不参与吞吐统计**，只用于检查事件结构：

| Trace 指标 | mini | MCore | 可支持的判断 |
| --- | ---: | ---: | --- |
| NCCL collective kernel 实例数 | 11,426 | 11,426 | mini 没有更多 collective 调用。 |
| 全部 CUDA kernel 实例数 | 145,650 | 176,114 | mini 不是因 kernel 数量更多而慢。 |
| 计时循环内显式 `synchronize` / `barrier` | 无 | 同一 benchmark 外壳 | mini 没有逐 update 强制同步。 |
| Nsight 下 12-update elapsed | 7.06 s | 16.69 s | capture 严重扰动了 MCore 路径，不能将 trace 的绝对时间用于解释正式的 18 ms/update 差异。 |

特别是，Nsight 导出的累计 NCCL kernel time 在 MCore trace 中为 31.49 s、mini 为
8.53 s，而 collective 次数相同；这与无 profiler 的正式吞吐方向相反。累计 kernel time
还跨两个 GPU 相加，并不等于 wall-clock。因此这两份 trace **排除了“mini 多通信、更多
kernel、逐步显式同步”这三个猜测，却不能证明 mini 慢在 NCCL，也不能给出唯一根因**。
更细的通信/计算重叠、小 kernel 排布、临时 copy 或框架调度只是待验证假设。

原始 report、SQLite 和 CSV 导出保留在 L20 归档；查看时应复制 `.nsys-rep` 后再用 GUI
打开，不能修改归档原件：

~~~text
diagnostics/20260821T1054Z-tp2-nsys-mini/
  tp2-mini.nsys-rep
  tp2-mini.sqlite
  cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
  cuda_api_sum_cuda_api_sum.csv
  cuda_gpu_trace_cuda_gpu_trace.csv
diagnostics/20260821T1054Z-tp2-nsys-mcore/
  tp2-mcore.nsys-rep
  tp2-mcore.sqlite
  cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
  cuda_api_sum_cuda_api_sum.csv
  cuda_gpu_trace_cuda_gpu_trace.csv
~~~

若要可靠归因，应在非 profiler 的 200-update 配对 benchmark 中加入低扰动的 CUDA Event
分段计时（forward、loss/vocab、backward、AdamW），再按五组配对报告每段的均值和差异；
不能从这两份全量 Nsight trace 的绝对耗时反推正式吞吐根因。

## mini 内部优化观察（不是跨框架结论）

早期 mini-only BF16 实验对 fused 与 unfused AdamW 做过 5 组交替配对，观测到
fused=True 约 **+17.08%** 吞吐。这可以说明项目经历了“profile 提出假设、改代码、
再重复测量”的优化闭环；但它不能替代跨框架语义等价，也不能解释当前跨框架差异的因果。

## 不应该再宣传的历史记录

早期 README 中的 1.6–2.4x 与 2.10x/2.28x mini 对 MCore 数字，使用过独立初始化、
独立随机输入或不同原生计算图的脚本。它们仅作为历史诊断保留在 evidence ledger，
**不能**证明框架性能优劣。

同样，identity-token 的 loss 曲线只是训练 wiring smoke test，不是训练质量、收敛速度
或泛化能力比较。

## 如何复现下一次受控实验

共享 artifact 构造、equivalence gate、benchmark runner 与 summary 都有对应脚本，不依赖
手写表格：

~~~bash
# 先运行 shared-contract parity gate
python3 experiments/validate_mcore_equivalence.py --help

# 再运行 paired runner；每个命令产生一个 immutable bundle
python3 experiments/run_fair_tp1_benchmark.py --help

# 发布前逐个校验 bundle，再汇总
python3 experiments/validate_run_bundle.py results/runs/<run-id>
python3 experiments/summarize_paired_results.py --help
~~~

遵守 [实验协议](experiment-protocol.md)：使用 clean source tree、阻止 active GPU、冻结
artifact 和配置、至少 5 组 ABBA/BAAB 风格配对，并将 profile capture 与吞吐计时分开。

## 要怎样扩大结论范围

1. 让共享权重的 numerical gate 在 BF16 下通过。
2. 将同一合同与 gate 扩展到 TP、PP 和多卡。
3. 对更大模型和生产相关的 optimizer/configuration 重复实验。
4. 使用固定的真实 train/validation split 与 held-out PPL 讨论训练质量。
