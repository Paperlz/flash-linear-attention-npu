# ChunkKdaFwd 第八轮 profiling：正常执行 PMU 与核间尾部

## 结论与当前任务方向

任务书的性能标准是最低要求，继续追求更低耗时。本轮完成 Case 250 的正常执行
PipeUtilization、Memory、ResourceConflictRatio 采集及部分核的时间采集，
获得可用于下一轮优化的设备证据。当前仍运行第五轮 wheel，本轮没有修改 kernel，
以下数值是当前基线的测量，不是新候选的加速收益。

- profiling 后原 ATK performance_device 复测成功：**768.506450 μs**，20 次采样，
  相对任务书 1323.737 μs 降低 **41.944174%**。
- 独立 PMU 三组中，ChunkKdaFwd 均值在 **682.607500–683.468750 μs**，
  与此前正常 ATK 的主 kernel 耗时相符。
- 每次调用采到的 4 个 AIC、8 个 AIV 样本之间，完成时间差平均 **51.554500 μs**，
  范围 46.658–55.413 μs。该模式未覆盖全部核心，不能当作完整核间分布。
- AIC 平均生命周期约 **652.935820 μs**，与主 kernel 的 683.468750 μs 相差
  **30.532930 μs**。假设每核总工作量不变且可以均摊，纯负载均衡的优化空间约为
  30 μs；这只是模型估算，不能直接宣称可获得 51.55 μs 的收益。
- 当前数据不足以把 AIC Scalar 活跃时间全部归因于地址计算，也不足以确认
  Prepare 内部具体哪一段等待最长。硬件指令轨迹未完成，不用于该结论。

## 输入、版本与采样口径

Case 250：B=1、T=1024、H=96、K=V=128、BSND、chunk_size=64，
Q/K/V=BF16、G=FP32、Beta=BF16，safe_gate=true、use_gate_in_kernel=true，
cu_seqlens=[0,1024]，seed=20260812。输入和公开调用复用现有
`benchmark_first_round.build_case_250_inputs`、`run_once`。

设备为 Ascend950PR，28 AIC / 56 AIV，采集记录频率 1650 MHz；
CANN 9.1.0-beta.3，Python 3.12.13，torch 2.7.1、torch_npu 2.7.1.post8。
安装 wheel 来源仍为第五轮 `round5_accuracy/opt_dist`。

| 采集 | 预热与采样 | ChunkKdaFwd 均值 / 标准差（μs） |
| --- | --- | ---: |
| PipeUtilization | 11234 次、10.000142 秒；20 次采样 | 683.468750 / 1.537732 |
| Memory | 11248 次、10.000633 秒；20 次采样 | 682.983050 / 2.157983 |
| ResourceConflictRatio | 11242 次、10.000372 秒；20 次采样 | 682.607500 / 1.873761 |
| 部分核时间 | 外部 msprof 记录 25 次，舍弃前 5 次 | 684.440500 / — |
| 正式 ATK 复测 | ATK 原预热和 performance_device 配置；20 次 | 768.506450 / — |

PMU 采集前预热至少 300 次且至少 10 秒，每次预热后同步；预热不在 profiler 窗口内。
每个 PMU 窗口都核对为 20 个 ChunkKdaFwd、20 个 Transpose。
独立公开 API 采集不包含 ATK executor 的输出有限值检查，只用于诊断。
正式任务书比较继续使用原 ATK 口径，不从测试里移除检查以制造性能收益。

Pipe 采集后最初分析脚本因 CSV 字段 `OP Type` 大小写错误而退出，错误发生在
设备采集结束后，已通过离线解析修正。该次日志也提示默认 schedule 在 RECORD
状态结束；实际目标设备任务计数为完整 20 次。后两组使用显式 20 次 schedule，
外部预热完成后开启采集；其“无内部 warmup”提示不代表未执行预热。

## 正常执行流水指标

以下是 20 次 `op_summary` 字段的平均值。各流水可以重叠，时间不可相加为关键路径。

| 执行单元 | 活跃时间（μs） | ratio |
| --- | ---: | ---: |
| AIC MAC | 105.484950 | 0.15435 |
| AIC Scalar | 284.730300 | 0.41680 |
| AIC MTE1 | 48.616850 | 0.07100 |
| AIC MTE2 | 198.593350 | 0.29070 |
| AIC Fixpipe | 217.663850 | 0.31865 |
| AIV Vector | 303.607600 | 0.44450 |
| AIV Scalar | 127.980100 | 0.18740 |
| AIV MTE2 | 107.139350 | 0.15690 |
| AIV MTE3 | 116.676250 | 0.17095 |

`cube_utilization(%)=95.53265` 不表示 MAC 算力利用率达到 95.5%。
已核对本机 CANN 的 `analysis/viewer/runtime_report.py:cube_usage`：
它由总 AIC 生命周期 cycles /（频率 × 核数 × task duration）计算；
MAC 活跃占比应看 `aic_mac_ratio=0.15435`。

Memory 原始均值：AIC L1 read/write 35.97370 / 17.98675 GB/s，
AIV UB read/write 16.00605 / 14.52055 GB/s。AIC main_mem read/write
0.50625 / 0.08020 GB/s，AIV main_mem read/write 0.11245 / 0.12085 GB/s。
这些是工具字段原值，未将其当作整卡总带宽，也未据此计算硬件饱和度。

ResourceConflictRatio 原始均值：
`aiv_vec_bank_cflt_ratio=0.15690`、
`aiv_vec_resc_cflt_ratio=1.36000`。
后者大于 1，保留原始计数比，不把它解释成“136% 的墙钟时间在等待”。

## 正式 ATK 设备任务分解

| OP Type | 20 次调用内任务数 | 每次调用时间（μs） |
| --- | ---: | ---: |
| ChunkKdaFwd | 20 | 683.612000 |
| Transpose | 20 | 8.311100 |
| IsFinite | 60 | 48.176800 |
| ReduceAll | 60 | 28.406550 |
| 合计 | 160 | 768.506450 |

算子自己的两个设备任务合计 691.923100 μs；ATK 有限值检查合计 76.583350 μs。
两项相加与 ATK 日志一致，不再将这部分差值归因为 launch/dispatch。

## 下一轮候选及代价

1. **优先设计 FwdH 的更细任务粒度。** 当前 ProcessAic/ProcessAiv 按完整 head
   循环；96 个 head 分给 28 个逻辑核，每核为 3 或 4 个 head，16 个 chunk 的
   状态递推属于 head 内串行依赖。可评估沿独立的 V 列切为两份、形成 192 个任务，
   最慢核为 7 个半 head。理想化工作量从 4 个 head 降为 3.5 个，预期收益是
   二十几微秒量级；必须同时核算重复 W/Post-WU、输入读取、小矩阵 MMAD 效率、
   L1/UB 地址和同步成本，收益不能直接按整体减半估算。
2. **Prepare 的阶段级定位继续保留为重点。** 全核 Scalar / Vector / Fixpipe
   都占有明显时间，但现有聚合 PMU 无法证明哪一项是最长依赖链。下一次阶段探针
   应围绕 score 生产、solve 和发布/等待边界设置有限计时点，并核对探针对总时长的
   扰动，然后只实验一个重排或复用变量。
3. 单纯将核数从 28 改为 24，仍有每核 4 个 head，同时增加 Prepare 每核任务数，
   缺乏缩短当前最慢路径的依据，暂不作为首选。

候选实施后应先通过任务书相关精度回归，再以相同 ATK 配置交替 A/B；
扩展精度研究不在本轮 profiling 内展开。任务书最低目标已满足不构成停止依据。

## 采集方式、产物与失败记录

实验目录：`round8_profile`。主要文件：

- `profile_case250_pmu.py`、`analyze_pmu.py`：采集与离线解析脚本。
- `pipe/`、`memory/`、`conflict/`：三组原始 CSV、trace、日志及采集元数据。
- `pmu_summary.json`：正常执行 PMU 汇总。
- `block_extrema/`、`block_extrema_summary.json`：部分核采集与后 20 次结果。
- `atk_final/`、`atk_final_summary.json`、`atk_final.log`：最终正式 ATK 复测。

部分核采集的已验证命令（环境按 ATK README 准备）：

```bash
msprof --task-block=on --task-time=l1 --ai-core=on \
  --aic-mode=task-based --aic-metrics=PipeUtilization \
  --output=<output_dir> \
  --application='python tests/atk/chunk_kda_fwd/scripts/profile_case250_kernel.py --iterations 25'
```

从 `device_0/sqlite/soc_log.db` 的 BlockLog 表按 op_summary 的 stream/task ID
关联目标 kernel，舍弃前 5 次，得到每次 4 AIC / 8 AIV 样本。物理 core_id 不是
源代码 GetBlockIdx 的直接替代，不能把采到的物理编号当成逻辑 head 分配编号。

本机 `--task-block=all` 首先要求额外的 `--optype`，添加参数后返回 1 且无完整输出，
因此未获得完整逐核覆盖；正式记录仅使用成功的 `--task-block=on` 数据。

另外一次 `--instr-profiling=on` 未正常完成；工具提示
`instr-profiling-freq is useless on this platform`，未生成完整可解释的指令报告，
部分 biu 数据只留作失败证据。未确诊根因，未修改 profiler 安装或尝试设备重启。
随后已确认该采集进程退出、NPU 无残留任务，并通过上述 ATK 复测。
