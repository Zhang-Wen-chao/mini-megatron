# mini-megatron

> 用可读的 PyTorch 代码实现 Megatron 风格训练的核心：张量并行（TP）、非交错 1F1B
> 流水线并行（PP）、数据并行（DP）和 BF16 AMP；同时保留测试和可审计的实验证据。

这不是 Megatron-Core 的替代品。这个仓库的价值是把核心训练机制压缩到可以完整阅读、
实际运行、自动测试和性能剖析的范围内。训练主链路的入口、模型、并行模块和 TP autograd
原语合计约 **830 行**。

## 这个项目的亮点

| 亮点 | 真正实现了什么 | 如何验证 |
| --- | --- | --- |
| **三维并行拓扑** | 每个 rank 都有 (dp, pp, tp) 坐标，TP、PP、DP 的 process group 显式构建。 | `parallel/process_groups.py` 可直接阅读，另有模型与并行原语测试。 |
| **真正的 1F1B，不是流程图** | PP 执行 warm-up forward、F/B 交替、P2P activation/gradient 回传和 look-ahead recv；默认支持 PP >= 2。 | CPU/Gloo 多进程测试验证 PP=2、PP=4 更新后的参数与单进程参考一致。 |
| **最小但完整的 TP 数学** | QKV/MLP-up 使用 column parallel；attention output/MLP-down 使用 row parallel，只在必须处 all-reduce。 | TP 与可微 all-reduce 测试；核心代码集中在 `parallel/tensor_parallel.py` 和 `comm/all_reduce.py`。 |
| **实验资产也是项目的一部分** | 实验运行生成不可变 bundle：命令、环境、日志、checksum，以及可选的原始 Nsight Systems 报告。 | 38 项测试覆盖实现和实验资产工具；已提交的 bundle 可做 checksum 校验。 |

## 正式性能结论：范围明确，而不是营销口号

目前最可信的跨框架比较故意收窄了范围：

| 控制条件 | 固定内容 |
| --- | --- |
| 硬件与并行 | 1 张 NVIDIA L20，TP=1，PP=1 |
| 模型 | 共享的 125M 无 bias GPT：12 layers、hidden 768、12 heads、FFN 3072、learned positions、pre-LN、GELU、无 dropout |
| 等价控制 | 相同转换后的初始权重（101 个 tensor）、相同的 230 个固定 next-token batch、FP32、standard unfused AdamW |
| 数值门槛 | 初始权重 max diff = 0；logits、最坏 gradient 与一步更新后的参数均在预先声明的阈值内 |
| 测量方法 | clean tree、空闲 GPU，5 组交替配对；30 warm-up + 200 measured steps |
| 吞吐 | mini **32,669.6 tok/s**；matching Megatron-Core custom-loop path **27,704.8 tok/s**；配对比 **1.179204x**（约 **17.9%**） |

它只支持这句话：**在上述精确条件下，mini 的匹配训练路径吞吐是 matching
Megatron-Core custom-loop path 的 1.179204x。** 它不代表默认 Megatron-Core、BF16、
fused optimizer、大模型、TP/PP/多卡扩展、生产训练负载或训练质量的通用结论。

完整方法、原始资产和限制请看：[实验协议](docs/experiment-protocol.md)、
[证据 ledger](docs/experiment-results-2026-08-17.md)、[benchmark 指南](docs/benchmarks.md)。

## 15 分钟读代码路线

1. [main.py](main.py)：构建并行拓扑，选择 PP 或非 PP 训练路径，然后执行 AdamW。
2. [parallel/process_groups.py](parallel/process_groups.py)：全局 rank 如何映射为 (dp, pp, tp)。
3. [parallel/tensor_parallel.py](parallel/tensor_parallel.py) 与 [comm/all_reduce.py](comm/all_reduce.py)：Column/Row 配对与其 backward 语义。
4. [parallel/pipeline_parallel.py](parallel/pipeline_parallel.py)：1F1B、P2P activation/gradient 回传与 look-ahead。
5. [tests/test_pipeline_1f1b.py](tests/test_pipeline_1f1b.py)：PP=2/4 的参数等价测试。

## 快速开始

需要：Python、带 CUDA/NCCL 的 PyTorch，以及 1–4 张 NVIDIA GPU。项目在 L20 上完成过
实验；Megatron-Core 只在运行 matching benchmark path 时才是可选依赖。

~~~bash
pip install -r requirements.txt

# 单卡 FP32
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1

# 单卡 BF16 autocast
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 --amp

# 两卡 Tensor Parallel
torchrun --nproc_per_node=2 main.py --tp 2 --pp 1

# 四卡 TP=2、PP=2；默认使用 1F1B
torchrun --nproc_per_node=4 main.py --tp 2 --pp 2 --schedule 1f1b
~~~

`serial` 是 legacy 的 PP=2-only 参考路径；请使用 `1f1b` 作为已接入的流水线调度。

## 什么已接入，什么还没有

| 领域 | 状态 |
| --- | --- |
| TP | 已接入：Column/Row linear 切分与 autograd all-reduce |
| PP | 已接入：非交错 1F1B、P2P activation/gradient 回传、PP >= 2 |
| DP | 已接入：每次 local optimizer step 前的 gradient all-reduce |
| AMP | 已接入：CUDA 设备上的 BF16 `torch.autocast` |
| Checkpoint / ZeRO-1 / sequence parallel / overlap helpers | 作为参考模块存在；**未接入** `main.py` 的端到端主链路 |
| Interleaved 1F1B、activation checkpointing、真实数据管道、MoE、CP/FSDP | 主链路尚未实现 |

这个边界是刻意说明的：文件存在不等于端到端能力已经接入；只有接入并经过测试的能力才会
标成“已实现”。

## 实验证据与 Nsight Systems 资产

每个受控 bundle 都保存命令、环境、stdout/stderr、metrics、manifest 与 SHA-256
checksum。两份 clean FP32 profile 的原始证据已版本化：

- `results/runs-clean-ad82d7e/20260818T003611Z-fair-tp1-fp32-clean-profile-mini/`
- `results/runs-clean-ad82d7e/20260818T003641Z-fair-tp1-fp32-clean-profile-mcore/`

每份都含原始 `.nsys-rep`、SQLite、CSV 导出、analysis JSON 和 checksum。引用前先验证：

~~~bash
python3 experiments/validate_run_bundle.py \
  results/runs-clean-ad82d7e/<run-id>
~~~

请先把原始报告复制到仓库外再用 GUI 打开；profiling 软件可能改写报告元数据。Git 中的
版本及其 checksum 才是原始证据。

## 仓库地图

~~~text
main.py                         训练入口与并行拓扑组装
model/                          GPT 组件与 loss
parallel/                       TP、PP 1F1B、DP 与 process-group 逻辑
comm/                           可微 all-reduce 与参考通信模块
experiments/                    可复现实验、parity gate、汇总、bundle 校验
tests/                          38 项实现与实验资产检查
results/                        已提交的 aggregate 与 clean evidence bundle
docs/architecture.md            数据流与设计决策
docs/benchmarks.md              当前结论、历史边界与复现方法
docs/experiment-protocol.md     一项性能结论的发布规则
~~~

## 测试

~~~bash
pytest -q
~~~

完整 clean L20 运行记录为 **38 passed / 11.28s**。测试包含模型/TP/训练检查、
CPU Gloo 下的 PP=2/4 1F1B 等价性、QKV mapping round-trip、bundle checksum 校验和
配对结果汇总规则。

## 深入阅读

- [架构与数据流](docs/architecture.md)
- [Benchmark 与复现](docs/benchmarks.md)
- [可信实验协议](docs/experiment-protocol.md)
- [L20 证据 ledger](docs/experiment-results-2026-08-17.md)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

## License

MIT — see [LICENSE](LICENSE).
