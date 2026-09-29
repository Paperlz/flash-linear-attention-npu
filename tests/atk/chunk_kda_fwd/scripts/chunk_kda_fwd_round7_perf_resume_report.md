# chunk_kda_fwd 第七轮记录：恢复性能基线与核对瓶颈

## 本轮范围

撤回第六轮尚未验证的 FP32 final_state 候选，保留第二轮 gk 精简写回、第四轮 W 回传 L1 批量化、第五轮 K/score 缓冲区生命周期修复。本轮没有新增 kernel 优化，不把环境恢复后的数值波动计为新收益。

prepare 和统一 kernel 入口已与第六轮开始前的第五轮快照逐字节一致；第六轮新增头文件及设计、测试增量已撤回。恢复后的 12 条补充用例 JSON 与第五轮原文件完全相同。实验候选和原始日志已另行留档，当前安装来源仍是已完成完整构建及验证的第五轮 wheel。

## Case 250 性能复测

- Ascend950PR，CANN 9.1.0-beta.3，ATK 26.9.8。
- B=1，T=1024，H=96，K=V=128，BSND，Q/K/V=BF16，G=FP32，Beta=BF16，chunk_size=64。
- safe_gate=true，use_gate_in_kernel=true，cu_seqlens=[0,1024]，使用原有 ATK 输入与执行器。
- 预热沿用 ATK 默认配置：最多 300 次、最长 10 秒；本次正式采样 20 次。
- 性能口径保持 ATK performance_device，即 op_summary 中全部被采集设备任务的 Task Duration(us) 总和除以 20。
- 本轮复测均值 **768.979350 μs**，标准差 **2.586090 μs**；相对任务书 **1323.737 μs** 降低 **41.908449%**，达到任务书的 5%–10% 性能目标。

| 设备任务 | 20 次调用的任务数 | 每次算子调用合计（μs） |
| --- | ---: | ---: |
| ChunkKdaFwd | 20 | 683.208000 |
| Transpose | 20 | 8.649050 |
| IsFinite | 60 | 48.524750 |
| ReduceAll | 60 | 28.597550 |
| 合计 | 160 | 768.979350 |

算子本体的两个设备任务合计 691.857050 μs；ATK 执行器对三个输出执行有限值检查，额外采集 77.122300 μs。该分解直接来自同一份 op_summary，合计与 ATK 报告完全一致。历史瓶颈报告把相近差值笼统归为 launch/dispatch 开销并不准确；后续比较继续保持原 ATK 口径，同时分别列出算子任务与检查任务，执行器不作变更。

## 成功 ATK 采样中的流水指标

以下均为 ChunkKdaFwd 的 20 次 MIX_AIC 记录均值。不同流水存在重叠，不能把各项忙时相加作为总耗时。

| 指标 | 均值（μs） |
| --- | ---: |
| aic_mac_time(us) | 105.356850 |
| aic_scalar_time(us) | 284.009900 |
| aic_mte1_time(us) | 48.589750 |
| aic_mte2_time(us) | 198.643850 |
| aic_fixpipe_time(us) | 217.750750 |
| aiv_vec_time(us) | 302.938400 |
| aiv_scalar_time(us) | 127.776800 |
| aiv_mte2_time(us) | 106.908800 |
| aiv_mte3_time(us) | 115.686900 |

## 候选方向与边界

当前 Task 的 blockDim=28，mixBlockDim=56。FwdH 的 AIC/AIV 都沿 batch × value-head 做 grid-stride，Case 250 的 96 个 head 使 0–11 号物理核各处理 4 个 head、其余 16 核各处理 3 个 head；每个 head 仍按顺序处理 16 个 chunk。此处是已核对源码和实测 blockDim 后得到的任务数量，不代替阶段计时。

原第一轮的阶段计时和本轮流水指标共同指向：Prepare 的标量下发/流水依赖，以及 FwdH 状态递推与 head 分配形成的尾部负载，仍是优先研究方向。已有第二、三轮实测不支持继续单纯削减被隐藏的 GM 写回或合并 W/U 发布。

简单把 blockDim 从 28 降到 24 并不能消除关键路径上的每核 4 个 head；还会增加 Prepare 的每核任务数，因此没有据此新增限核分支。更细的 FwdH 任务粒度需要先评估重复 W 计算、矩阵效率、状态分区与同步成本，再决定是否实验。

本轮性能目标已经满足，不继续无依据地枚举 kernel 变体。原始 48 条 ATK 精度和 Case 250 50 次确定性沿用第五轮正式结果；本轮只做恢复核对和性能采集。补充泛化的 final_state 问题仍以第五轮报告为准，未宣称全量泛化验收通过。

## 复现

准备 CANN、Python 和已安装 fla_npu OPP 环境后执行：

```bash
CASE_START=0 CASE_END=1 ATK_OUTPUT_ROOT=<output_dir> \
bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd -npu_device_id=0 -soc=ascend950 -scope=performance
```

核对 profile 目录中的 op_summary，按 OP Type 汇总 Task Duration(us) 并除以 20；ChunkKdaFwd 和 Transpose 分别应有 20 个 Task，IsFinite 和 ReduceAll 各有 60 个 Task。

第五轮代码及验证结论见 [第五轮报告](chunk_kda_fwd_round5_kg_lifetime_report.md)。

## 附加采集的失败与恢复检查

独立 msopprof 的三次采集（kernel replay 的 BasicInfo/PipeUtilization，增加 Occupancy，以及 application replay 加 Occupancy）均在 operand record 路径报错，包含 `Run operand record func failed` 和 NPU 错误 507015。CLI 最终退出码为 0，但子应用失败，不能作为采集成功的依据。application replay 只采 BasicInfo/PipeUtilization 的后续尝试返回 1，未生成完整报告。

这些采集产生的部分 per-core 数据仅作原始失败记录保留，不用于验收结论；不能声称仅切换 replay 模式便已解决。已确认对应采集进程退出、设备资源释放，未重启或重置设备。

随后使用同一第五轮 wheel、相同原始 Case 250 和原 ATK 命令重新测试，20 次采样成功，性能为 **769.429050 μs**。两次成功 ATK 均值为 **769.204200 μs**，相对任务书标杆降低 **41.891463%**。结果证明当前正常 ATK 性能路径仍可执行；单独的 msopprof 插桩问题留待需要该工具时再定位，本轮不扩大工具排障范围。
