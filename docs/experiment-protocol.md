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
不同顺序消耗 RNG。除非显式传入 `allow-dirty`，runner 会拒绝工作树存在未提交改动的运行；
允许该选项时，manifest 会保留 `source_tree_clean=false`。默认不应引用尚未固定代码快照的
此类运行；若实验先发生，后续可将实际运行的关键源码、runner、contract 与测试文件逐一按
内容 hash 核验并固定到一个提交，则可按报告的精确合同引用，并且必须公开原始状态与核验方法。

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

## 125M 多卡 campaign：档案结构与发布门槛

125M 扩展实验只覆盖 `TP=2,PP=1,DP=1`、`TP=1,PP=2,DP=1` 与
`TP=2,PP=2,DP=1`；已有 `TP=1,PP=1,DP=1` 是单卡锚点。不要把这套结果泛化成
更大模型或所有 Megatron 配置。每个新增拓扑都使用同一份预先冻结的协议：FP32、
`B=4`、`S=512`、每个 optimizer update 8 个 micro-batch（16,384 tokens）、30 个 warm-up
update 与 200 个 measured update。这样既让 PP 的 1F1B 具有足够 micro-batch，又让三个
多卡拓扑的 tokens/update 完全相同。

L20 上为一次 campaign 创建**只追加**的归档目录，例如：

~~~text
/mnt/storage01/zhangwenchao02/repos/mini-megatron-test/evidence/
  fair-125m-parallel-20260821/
  ├── campaign.json                 # 一次写入：合同、门槛、固定执行顺序
  ├── artifacts/                    # full source weights、mini/MCore shards、固定 batches、manifest
  ├── parity/                       # 每个拓扑的 logits/grad/one-step update report
  ├── benchmarks/                   # 10 个 immutable run bundle/topology（5 对）
  ├── profiles/                     # 单独的 Nsight bundle；绝不参与吞吐统计
  ├── reports/                      # paired aggregate、可发布结论与范围
  └── ledger/                       # 每份证据的路径、大小与 SHA-256，只追加
~~~

初始化命令会把模型合同、阈值、拓扑矩阵、`mini,mcore,mcore,mini,...` 的 10 次固定
执行顺序写入 `campaign.json`。之后只用 `record` 追加一份证据的地址和 SHA-256；不能
原地覆盖旧报告。

~~~bash
python3 experiments/fair_parallel_campaign.py init \
  --campaign-id fair-125m-parallel-20260821 \
  --campaign-dir /mnt/storage01/zhangwenchao02/repos/mini-megatron-test/evidence/fair-125m-parallel-20260821

python3 experiments/fair_parallel_campaign.py record \
  --campaign-dir /mnt/storage01/zhangwenchao02/repos/mini-megatron-test/evidence/fair-125m-parallel-20260821 \
  --kind numerical_parity --topology tp2-pp1-dp1 \
  --source /mnt/storage01/.../parity/tp2-pp1-dp1/report.json

python3 experiments/fair_parallel_campaign.py validate \
  --campaign-dir /mnt/storage01/zhangwenchao02/repos/mini-megatron-test/evidence/fair-125m-parallel-20260821
~~~

校验器会验证每一条登记的 SHA-256，并列出每个拓扑是否缺少 `artifact`、
`numerical_parity`、`benchmark_summary` 或 `profile`。只有四类都已登记且底层 bundle
自身通过 `validate_run_bundle.py` 时，才可以把这个拓扑写入结果文档。仓库提交的是
campaign 配置、ledger 副本、aggregate、命令、环境快照、校验和和可移植 CSV；L20
归档保留权重 shard、输入、全部 bundle 以及 `.nsys-rep/.sqlite` 原件。Git 中的原始 profile
只能从未打开的副本提交；GUI 查看必须复制到仓库外，避免修改报告元数据。

**证据链完整不等于原始数值门槛通过。**ledger 的每项证据带有 `claim_status`：只有四类
证据都标为 `ordinary`，才允许无条件性能结论。若使用事后探索性校准，必须标为
`conditional_exploratory`，并在汇总、结论和文档中同时保留原始 gate 未通过的事实、
校准来源、适用范围和不支持的主张。此时 ledger 可以显示“证据链完整”，但只能形成条件性
结论；不能因文件齐全就把后验校准改写为原计划门槛已通过。

失败也必须保留，但不得混入结论：为每一次 smoke、接口错误或被替换的方案保存
`diagnostics/<timestamp>-<topic>.json`（命令、stdout/stderr、触发条件、根因、后续修复
commit），并用 `record --kind diagnostic` 登记 SHA-256。`diagnostic` 永远不会满足上面的
发布门槛。这样项目能保留真实踩坑经验，同时不会把失败的临时实现伪装成公平实验资产。

`source_tree_clean=false`（通过 `--allow-dirty` 创建）只记录运行开始时工作树有尚未提交的
代码；它不表示 GPU、输入或测量结果被污染。默认仍应先提交 runner 与 workload，再运行实验。
若实验先发生，后续可将实际运行的关键源码、runner、contract 与测试文件逐一按内容 hash 核验
并固定到一个提交；报告必须同时保留原始 `source_tree_clean=false` 元数据、核验方法和固定
commit。满足这些条件时，该 bundle 可以支持其精确合同内的结论；新的 clean-tree 独立复跑是
增加重复性的推荐证据，而不是把已有受控数据自动判为无效的条件。

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
