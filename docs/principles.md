# 设计原则

> mini-megatron 的目标是把 Megatron 风格训练的核心机制做成一个可读、可运行、
> 可测试、可审计的教学实现；不是以较少代码替代生产训练框架。

## 一、教学完整性优先于功能数量

项目选择了一条明确边界：把 TP、non-interleaved 1F1B PP、DP 和 BF16 AMP 接入同一条
主训练链路，并让读者能从入口一路跟到通信、反向和 optimizer step。

约 830 行的 wired training path 不是“行数越少越好”的宣传，而是一个约束：

- 太少会把 P2P gradient 回传、TP autograd、DP 同步或 pipeline schedule 隐藏掉；
- 太多会让读者无法在一两次阅读中建立完整心智模型；
- 只保留解释核心机制所需的代码，生产能力则明确标为未接入或参考实现。

## 二、纯 PyTorch 是为了把机制露出来

| 选择 | 目的 |
| --- | --- |
| PyTorch 原生 module 和 distributed API | 让 forward、backward、collective 和 P2P 通信直接可见 |
| PyTorch SDPA | 复用框架选择合适 attention kernel 的能力，不把注意力从并行机制转移到手写 kernel |
| 不依赖 HuggingFace / Accelerate | 避免高层封装遮蔽 rank、group、shard 和通信时序 |
| 少量 reference modules | 让 ZeRO-1、sequence parallel、overlap 等概念可以被阅读，但不把“文件存在”误称为端到端能力 |

纯 PyTorch 的代价也很清楚：它没有生产 Megatron 的大量可扩展性、容错、调度与性能能力。

## 三、实现一条正确的 PP 反向链路比画 PP 图更重要

流水线 stage 之间不能共享 PyTorch autograd 图。前一 stage 发送 activation 后，后一
stage 必须将收到的 tensor 作为新的图根计算 activation gradient，并 P2P 回传；前一
stage 再以该 gradient 调用本地 backward。否则前一段模型不会获得正确梯度。

在此基础上，默认 non-interleaved 1F1B 将 schedule 分成 warmup、forward/backward
交替和 drain，并用 look-ahead recv 尝试重叠通信与计算。项目以 PP=2 与 PP=4 的
CPU/Gloo 多进程参数等价测试验证它和单进程参考的更新结果一致。

## 四、性能结论必须先满足等价，再谈速度

“名义模型大小相同、同一天运行、吞吐公式相同”不足以构成公平跨框架实验。当前协议要求：

1. 固定模型合同、初始化/转换后权重、token/label、mask、optimizer、precision 和步数；
2. 在计时前比较 logits、gradient 和一步更新后的参数，并记录阈值；
3. 在空闲 GPU 上做至少五组 ABBA/BAAB 风格交替配对，公布每次结果而非最快值；
4. 保存命令、环境、原始日志、manifest、checksum 和 profile 资产；
5. 将 profile 用于解释，不把 profile elapsed time 当作吞吐结果。

因此，当前能够引用的跨框架结论只有：在共享 125M GPT、固定 batch、FP32、TP=1/PP=1
和 standard AdamW 的 L20 合同中，mini 相对 matching MCore custom-loop path 的配对
吞吐比为 1.179204x。旧的 1.6–2.4x 与 2.x 数字来自不满足该协议的历史脚本，不能作为
性能优劣结论。

## 五、验证范围要和能力范围匹配

测试覆盖模型/TP/训练一步、PP=2/4 的 1F1B 参数等价，以及实验资产的 checksum、QKV
layout conversion 和配对统计。clean L20 会话的完整记录为 38 passed / 11.28s。

但这不等于项目已覆盖生产训练：interleaved 1F1B、activation checkpointing、真实数据
管道、ZeRO-1 主链路、sequence parallel 主链路、overlap、MoE、FSDP/CP 等仍未接入。
相应地，当前实验也不支持 BF16 parity、多卡公平对比、大模型、长训练收敛或 held-out PPL
结论。

## 六、何时应该升级而不是继续堆功能

当需求变成 1B+ 大模型、多机、MoE、真实数据训练或生产级容错时，应当把它视为新的
工程问题：要么使用 Megatron-Core / DeepSpeed 等成熟框架，要么单独建立 production
分支并补齐系统设计、测试和实验协议。教学主线不应悄悄膨胀成一个未经验证的生产框架。

## 七、证据优先的维护规则

- 新性能数字先进入带 checksum 的 run bundle，再写文档；
- 源码不 clean、GPU 不空闲、缺少数值等价或少于五组配对的运行，只能做探索记录；
- 原始 Nsight 报告是证据资产，GUI 打开前先复制到仓库外，避免改写版本化文件；
- 一旦发现旧结论控制条件不足，应保留追溯记录，但从首页与正式结论中撤回。

具体执行步骤见 [experiment protocol](experiment-protocol.md) 与
[benchmark guide](benchmarks.md)。
