# chunk_kda_fwd 第九轮：FwdH 尾部 V 切片实验

日期：2026-09-14。结论：**未获得提速，候选已撤回，保留第五轮版本。**

三组交替 A/B 的 ATK device perf 均变慢，均值从 **771.26125 μs**
增至 **777.40962 μs**，退化 **0.797%**。恢复第五轮 wheel 后复测为
**771.14440 μs**。本轮完成了实现、构建、边界比较、性能复测和撤回，
不把已达到任务书最低目标作为停止性能探索的依据，也不把本候选列为可交付优化。

## 1. 根据第八轮报告选择的变量

第八轮指出：96 个 head 分配给 28 组 AIC/AIV，存在3/4个head的负载差。
部分核采样的完成时间差约51.55 μs；平均核寿命与任务时长差约30.53 μs。
这些数值是候选方向的依据，不等于可直接兑现的收益。

本轮保留前84个head的原128列实现，仅将末12个head沿V维拆成24个64列任务。
一般条件为 N=B*HV、C=prepareUsedCoreNum、Q=floor(N/C)*C、R=N-Q：
只有0<R且2R<=C才拆尾部，否则保持原路径；不是Case250专用分支。

每个半任务由同一AIC和两个AIV完成其全部chunk递推。
K归约维度、BF16舍入和FP32累加不变，W仍为128列。
U、state、vNew、输出仅处理指定V列，GM行距仍为128。
前端、公开ABI、workspace容量和现有四槽深度保持原样。
相比将所有head拆分，额外W/Post-WU处理限于尾部；
Case250的FwdH矩阵乘理论计算量增加约1.39%，每chunk任务控制次数则增加12.5%。

代码改动限定在私有FwdH头文件和arch35调度入口。
全宽实例到半宽实例显式继承待消费的STATE_FREE/VNEW_FREE，
避免将跨head的mode2通知计数误清零。设计、地址图和同步自审保存在候选设计快照。

## 2. 构建和测量条件

- Ascend950PR，28 AIC/56 AIV，1650 MHz；CANN 9.1.0-beta.3。
- Python 3.12.13、torch 2.7.1、torch_npu 2.7.1.post8、ATK 26.9.8。
- 第五轮 wheel 为A，本轮完整单算子范围 wheel 为B；每次更换后重新启动Python。
- Case250：B=1,T=1024,H=HV=96,K=V=128,CS=64；原始输入、seed和ATK配置不变。
- 先完成构建，再执行正式三组A/B；每次20个有效设备样本。
- 样本均为1个ChunkKdaFwd、1个Transpose、3个IsFinite和3个ReduceAll。
  ATK总口径包含检查任务；不能把它和只含融合kernel的时间直接相减当作launch开销。

构建使用 `FLA_NPU_OPS=chunk_kda_fwd`、`FLA_NPU_SOC=ascend950`、
`OPS_CPU_NUMBER=4` 及 `python -m pip wheel --no-build-isolation --no-deps .`。
保持WebSocket执行会话的完整构建从10:56:42至11:22:41 UTC，成功生成wheel。
此前一次nohup后台进程被回收，没有完成标记；OOM计数为0，不将该次计为构建成功。

## 3. 性能结果

单位：μs。变化为(B-A)/A，正值表示变慢。

| 配对 | 第五轮 A | 尾部切片 B | 变化 |
| --- | ---: | ---: | ---: |
| 1 | 771.48305 | 778.94090 | +0.967% |
| 2 | 771.73575 | 777.23250 | +0.712% |
| 3 | 770.56495 | 776.05545 | +0.713% |
| 均值 | 771.26125 | 777.40962 | +0.797% |

非配对初测为769.98030→780.87710 μs，仅作记录，不替代上表。
恢复第五轮后的771.14440 μs与A组一致，相对任务书1323.737 μs降低
41.745%。

下面的PMU均来自正式A/B各60个融合kernel样本，单位μs：

| 指标 | A | B | B-A |
| --- | ---: | ---: | ---: |
| 融合 kernel | 684.352 | 690.276 | +5.924 |
| AIC scalar | 284.499 | 309.685 | +25.187 |
| AIC MAC | 105.535 | 104.811 | -0.724 |
| AIC MTE1 | 48.668 | 49.130 | +0.462 |
| AIC MTE2 | 199.022 | 198.879 | -0.143 |
| AIC Fixpipe | 218.157 | 211.534 | -6.623 |
| AIV VEC | 303.389 | 297.758 | -5.631 |
| AIV scalar | 127.916 | 131.225 | +3.308 |
| AIV MTE2 | 106.713 | 109.625 | +2.912 |
| AIV MTE3 | 116.097 | 113.650 | -2.447 |

融合kernel样本标准差：A=1.929 μs，
B=2.535 μs。以上pipe活动可能重叠，不能相加解释总耗时。
AIC icache miss rate原始字段均值约0.002→0.00397，只是伴随变化，未据此单独判定瓶颈。

辅助核采样沿用第八轮成功的命令：

```bash
msprof --task-block=on --task-time=l1 --ai-core=on \
  --aic-mode=task-based --aic-metrics=PipeUtilization \
  --output=<采样输出目录> \
  --application='python tests/atk/chunk_kda_fwd/scripts/profile_case250_kernel.py --iterations 25'
```

丢弃前5次，分析后20次；每次仅有4个AIC和8个AIV采样点。
采样完成差为52.8263→48.61165 μs，仅缩短4.21465 μs，
标准差分别1.83165和3.43535 μs。物理core ID不能直接当作逻辑task ID，
此结果不代表所有核。before辅助采样时CPU仍在编译；正式A/B表及其PMU均在编译结束后取得。

解释：这份实现减少了部分VEC/Fixpipe活动，但AIC scalar增加约25.19 μs，
同时总kernel变慢；数据与更多半任务带来的控制/同步工作相符。
不能把“计算量减半”视为“head耗时减半”，也不能仅凭本轮PMU定位某个具体wait。
本轮排除的是当前尾部切片实现，而非证明所有负载均衡方法都无效。
后续性能方向应保留完整V任务粒度，优先减少每chunk的标量调度和同步/发布次数。

## 4. 结果比较与范围

直接复用现有ATK executor生成输入，将候选的全部公开输出与第五轮保存输出逐位比较：

| 输入 | 比较结果 |
| --- | --- |
| 原始Case250 | 全部逐位一致 |
| 原始Case251，导出中间量 | 全部逐位一致 |
| H=HV=12,T=320,NTD，initial/final state，纯半宽任务 | 全部逐位一致 |
| B=2,HK=8,HV=16,T=320,BNSD，initial/final state及中间量 | 有差异 |
| H=HV=43,T=320，尾部不拆分 | 全部逐位一致 |
| H=HV=28,T=320，完整一波 | 全部逐位一致 |

跨batch输入的attn_out、final_state、vNew、h不一致；输出均有限。
重新安装第五轮、重新生成同seed输入后，第五轮自身也无法与先前保存输出逐位一致。
候选与初始第五轮的attn_out最大差约0.0029831，第五轮重复间约0.0007076。
因此存在基线稳定性问题，不能将全部差异只归因于切片，也不能宣称候选已通过该边界。
根因尚未确定，保存复现数据；本轮没有修改golden、值域或阈值。

逐位比较不替代ATK CPU golden验收。
已准备12个原始dense受影响用例，但因性能淘汰撤回候选，未继续执行其完整CPU精度验收，
也未扩展此前的FP32精度课题。原48例历史通过结论未冒充本轮结果。

## 5. 恢复与证据

FwdH头文件、arch35入口和正式design.md已按修改前快照逐字节恢复。
第四轮W发布改动、第五轮kg修复及其他既有工作区变更均保留。
本轮临时validation文档归档到实验目录；正式源码不含V切片候选。
安装来源已确认回到第五轮wheel，恢复后的ATK性能任务1/1成功。
未提交或推送代码。

工作区外的 `round9_vslice` 实验目录保存：

- `fwd_h.before.h`、`impl.before.h`、`design.before.md`、`baseline.patch`；
- `fwd_h.candidate.h`、`impl.candidate.h`、两个candidate.diff、
  `design.candidate.md`、`validation.candidate.md`；
- 候选wheel、构建及安装日志；
- `measure_perf.sh`、`analyze_perf.py`、A1/B1至A3/B3原始CSV、`perf_summary.json`；
- `check_boundaries.py`、保存的第五轮输出、候选比较和跨batch复现日志；
- `analyze_blocks.py`、before/after的CSV与BlockLog数据库；
- `evidence.json` 汇总全部测量来源和结果。

本轮正式留存的是实验报告与证据，候选性能代码未采用。
