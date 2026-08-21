# 可信实验协议

本文规定 mini-megatron 的实验能够声明什么，以及证据必须如何保存。README 和 benchmark
中的每一个新数字都必须遵守本协议。既有表格只是历史背景，不能单独支撑新的结论。

## 结论边界

受控对比应使用如下表述：

> 在已记录的硬件、软件、模型、精度、batch、sequence length 和并行配置下，
> mini-megatron 相对仓库中的 Megatron-Core custom-loop baseline 达到了所报告的吞吐。

不能把它写成“mini-megatron 比 Megatron 快”。custom loop 只隔离了一条窄的实现路径，
没有覆盖完整生产功能、大模型、多机运行或完整训练 recipe。profile 只能解释瓶颈，不能替代
吞吐测量。

## 发布结果前必须具备的证据

1. **语义等价。**使用 canonical checkpoint 和固定 token，与未切分 reference 比较
   logits、loss、gradient，以及一次 optimizer update 后的参数；记录阈值与精度。
2. **配置等价。**冻结 architecture、初始化/checkpoint、inputs/labels/mask、global batch、
   update count、token budget、TF32/BF16、optimizer、scheduler、clipping 和软件版本。
   任一字段不同都意味着非等价，必须公开说明。
3. **配对重复。**在空闲主机上至少运行 5 组独立 ABBA 或 BAAB 配对。公开每一次复现、
   mean、median、sample standard deviation、min/max 和 paired speed ratio；不能只发布最快一次。
4. **将扩展性与训练质量分开。**TP/PP/DP scaling 与 memory 要和单卡吞吐分开报告；若要讨论
   学习质量，应使用固定的真实 tokenized train/validation split 与 held-out PPL。identity loss
   仅是 wiring smoke test。
5. **profile 单独保存。**保留原始 Nsight 报告、SQLite、CSV、精确命令和分析规则；绝不把
   profile 的 elapsed time 当作吞吐结果。

## Run bundle 与原始 Nsight 证据

run_experiment 脚本执行一条命令，并生成一个不可变 bundle：

~~~text
results/runs/<timestamp>-<name>/
├── manifest.json       # scope、命令、计时、metrics、return code
├── environment.json    # git 状态、packages、GPU/topology、CUDA/NCCL 环境
├── command.txt
├── stdout.log / stderr.log
├── metrics.json
├── checksums.sha256
└── profile/            # 仅在 --profile 时生成
    ├── rank0.nsys-rep
    ├── rank0.sqlite
    ├── cuda_gpu_kern_sum.csv
    ├── cuda_api_sum.csv
    ├── cuda_gpu_trace.csv
    └── exports.json
~~~

nsys-rep 和 sqlite 是源证据，应放入 Git LFS 或不可变实验归档，并在 aggregate report 中记录
URI、文件大小和 SHA-256。manifest、命令、环境、checksum、CSV 与人类可读分析应保存在 Git。
绝不能原地替换旧 bundle。

每一个固定输入 checkpoint、token 文件或数据 shard 都应通过 artifact 选项传入，使 manifest
记录其字节大小和 SHA-256。相比只记录 random seed，这更可靠，因为不同框架的构建过程可能以
不同顺序消耗 RNG。除非显式传入 allow-dirty，runner 会拒绝 dirty source tree；允许 dirty 的
运行会在 manifest 中标记为 non-clean，不能作为主要发布证据。

kernel-summary analyzer 会在 profile/analysis.json 中记录 CSV SHA-256、regex 分类规则和所有
未匹配 kernel 时间。百分比是 kernel time，不是 wall-clock。除非有可独立审计的归因规则，
不能把通用 elementwise kernel 标为 unfused AdamW。

信任一个 bundle 前，运行：

~~~bash
python3 experiments/validate_run_bundle.py results/runs/<run-id>
~~~

## 在共享 L20 主机上的执行规范

宿主机使用 nerdctl container runtime；Docker 不是实验入口。容器配置的 workdir 已过期，
必须使用 -w /：

~~~bash
ssh l20 'nerdctl exec -w / zhangwenchao-megatron /bin/bash -lc "cd /mnt/storage01/zhangwenchao02/repos/mini-megatron-test && <command>"'
~~~

使用 NCCL_SHM_DISABLE=1；多卡运行另加 CUDA_DEVICE_MAX_CONNECTIONS=1。当前交接细节见
dev-guides/local-to-l20-handoff.md。

每条命令前先确认目标 GPU 空闲：

~~~bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader
~~~

runner 会执行同样的 preflight；除非显式允许 active-GPU override，否则发现 compute process
就会拒绝启动。override 运行可能被污染，必须排除在受控结果之外。

一次只运行一条命令。框架对比应交替执行（mini -> megatron -> megatron -> mini），而不能
一次跑完某一框架的全部样本；每条命令对应自己的 bundle。

source_tree_clean=false（通过 --allow-dirty 创建）的 bundle 仅是临时证据。先提交 runner 与
workload，再在 clean tree 上重跑配对实验，才可以更新 README 或 release 中的主结论。

一次 benchmark replicate 的例子：

~~~bash
python3 experiments/run_experiment.py --name mini-125m-tp1-fused-r01 \
  --tag variant=mini --tag pair=01 --tag condition=125m-s512-b4-bf16 \
  --artifact artifacts/canonical-125m.pt --artifact data/fixed_tokens.bin -- \
  torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
  --num-steps 200 --warmup-steps 30 --micro-batch-size 4 --amp --fused
~~~

profile 示例；它不是吞吐样本：

~~~bash
python3 experiments/run_experiment.py --profile \
  --name mini-125m-tp1-fused-profile-r01 -- \
  torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
  --num-steps 40 --warmup-steps 10 --micro-batch-size 4 --amp --fused
~~~

多 rank profile 时，为每个 CUDA rank 启动带唯一输出前缀的 nsys profile wrapper。不能只
profile torchrun launcher 后就假定它的子进程已经被捕获。

完成至少 5 组配对后，生成 aggregate，不能挑选最佳样本：

~~~bash
python3 experiments/summarize_paired_results.py --results-dir results/runs \
  --left mini --right megatron --output results/aggregates/125m-tp1.json
~~~
