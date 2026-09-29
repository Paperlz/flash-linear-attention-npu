# R31 交付状态（2026-09-29）

本分支保存 R31 Fix5 完整包对应源码；当前尚未通过全部验收。

- 固定 20 例（250–269）：20/20 通过；代表性泛化 20 例：15/20 通过。
- 使用 AscendOpTest 原始 compare_default，逐公开输出比较 CPU FP64 标杆；未修改判据或删除失败记录。
- 失败输出：3554 Aqk；5114 attn_out；5123 attn_out/h；5168 attn_out/h/v_new；5249 attn_out/final_state/h/v_new。
- Case250，Ascend950PR，完整包 100 次采样：主核 557.11428 µs，标准差 4.38254 µs；转置 9.01779 µs，合计 566.13207 µs。任务书 1323.737 µs 为历史标杆，未在同环境配对重跑。
- 完整 wheel 已构建并安装核对。完整包确定性、MSS 成功报告和共享 GDN 回归尚未完成；Fix3 的 19 例各 100 次确定性通过仅适用于当时版本。
- 按已确认口径保持参考代码接口：Aqk/Akk、w/u 等返回 q 对应的 BF16/FP16；任务书 §2.2 的 FP32 描述与参考实现不同，final_state 保持 FP32。
- 历史 465 例诊断中，10 例有 21 个输出失败；原始证据随本地交付包保存。当前默认集为固定 20 例加泛化 20 例，不代表穷举完整输入范围。

复现入口见仓库 tests/atk/chunk_kda_fwd/README.md。以下内容保留开发历史；roundXX 路径指当时的归档实验目录，当前结论以本节为准。

## 历史开发验证

## 输入、路由与基线
用户要求继续第八轮 profiling 后的优化，并通过云端 Codex exec-server 执行。
当前保留第四轮 W 发布和第五轮 kg 生命周期修复；第九轮 V 切片已撤回。
执行本次用户提供的 AGENTS.md 与本地 docs/agents/03-05 当前 V2 流程。
远端设计尚未记录规则版本，本轮先对照源码记录实际阶段，再更新受影响详设。
参考 CHUNK_DEPENDENT V1，来源 commit a14961e9dfe8c8264768d67fb447101873da4ecf；
仅采用顺序依赖、地址所有权和同步闭环的检查方法，保留本算子现有调度。

Case 250：B=1,T=1024,H=HV=96,K=V=128,CS=64,BF16,BSND，
原始模型输入与 seed=20260812；Ascend950PR，28 AIC/56 AIV。
当前安装来源 round5_accuracy/opt_dist，基线源码快照 round10/op_kernel.before。
性能统一记录 msprof op_summary Task Duration(us)：
1个 ChunkKdaFwd + 1个Transpose 为算子设备总时间；
并继续保留原 ATK 全任务合计，含3个IsFinite与3个ReduceAll。
正式性能每组20样本，A/B同输入、同配置、串行执行，三组交替。
目标是相对当前第五轮重复基线获得超过测量波动的收益，1323.737us仅是任务书旧标杆。
准确性保持现有 CPU FP64 golden、模型值域和混合容差。
首次误将case ID 250用作列表范围，未产生精度报告，不算通过；
已改为列表索引0:2，对应Case250/251。

## 阶段探针设计（仅诊断构建）
记录 arch35 Run 进入、Prepare返回、现有SyncAll后、FwdH进入、各head返回及FwdH返回。
GetSystemCycle只增加有限标量读，不添加同步，不更改原数学计算或生产消费顺序。
每个AIC/AIV独占128字节记录；AIC索引0..C-1，AIV索引C..3C-1。
使用dense FwdH路径未消费的outputScratch区，最大3*C*128字节，
该区既有分配2*B*T*HV*V*sizeof(float)，目标Case远大于记录容量。
仅在useDenseFwdH为真时写记录，结束后一次性写回。
时间在独立scalar数组中暂存，GM记录只在FwdH完成后发布。
同一核首个至第4个head的完成点用于确认3/4head任务实际执行。
探针kernel的msprof总耗时与未插桩基线对比，量化扰动；
阶段时钟先保留原始ticks，并与设备总耗时核对频率，不能使用AIC频率1650MHz代换。
探针用于阶段归因，最终性能必须撤去探针，并以正常构建A/B验收。

## 待验证方向
若FwdH处于关键路径，检查同一64x256 Post-WU MMAD结果拆为W/U
两次Fixpipe、两次mode4 ready/free的必要性；暂不采用候选。
其他备选为独立gate读取前移。每次只实验一个主要变量。

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 250/251 | 基线 | 第八轮PMU、九轮失败 | 保持第五轮 | 本轮复测中 | ATK 769.21625us；主kernel 682.4145us | 待探针 | 未插桩 |
| 250 | 待实测 | 阶段归因尚缺 | 有限阶段时钟探针 | 对比公开输出逐位一致 | 待测 | 待测 | 诊断产物 |

## 采用后的验证计划
首先核对Case250/251公开输出、必要边界；之后完整48条ATK CPU golden、
确定性和mssanitizer；目标性能三组A/B，检查每组任务计数与瓶颈指标。
若候选变慢或准确性未过，保留报告并恢复本轮修改前文件和第五轮安装产物。
不改公开接口、标杆、输入值域或容差。

## 阶段定位与实验门禁结果
Case250/251本轮CPU golden 2/2通过；探针公开输出与第五轮逐位一致。
GetSystemCycle对应DAV3510的get_system_cnt，设备info.json.0报告hwts_frequency=1000MHz，
不能使用旧A2文档50MHz。初次换算结果标为无效，原始ticks保留。
同窗口20次/84核全覆盖：trace span=679.5691us，msprof=683.94195us，
逐次差3.048-5.382us。无采数clone的探针msprof=682.00815us，对照基线682.4145us。
AIC前端均值446.719825us（Prepare429.551709us，既有barrier13.431746us），
后端3head=182.3692us、4head=228.243358us。前端主要占比约65%，后端处于串行关键路径。
因此选定W/U成对发布实验：不减少head/不拆V，不减少MMAD或Fixpipe字节，
只减少一次每chunk的direct ready/free及M_FIX/FIX_M交接。
候选完整地址/生命周期/同步评审见实验目录paired_publication_design.md。
状态、W与Vnew的mode2数量不变；direct区域原有32KiB足够两个compact半片。
理论最多64次/最慢核的交接可省开销须大于约2us重复波动，收益需要实测确认。
下一诊断版本还记录实际chunk、slot使用次数、最后task/chunk、DIRECT_READY/FREE计数，
核对跨head与尾部闭环；正式性能测量前撤除全部探针。

## 候选诊断验证
Case250/251全部公开输出与第五轮逐位一致。
20次、28逻辑核组的AIC/AIV实际记录一致：
4head核执行64chunk，DIRECT_READY=320、DIRECT_FREE=324，四槽各使用16次；
3head核执行48chunk，DIRECT_READY=240、DIRECT_FREE=243，四槽各使用12次。
最后task/chunk及head完成点均符合运行时grid-stride；额外FREE来自每head初始化/尾部drain。
证明成对发布、四槽回绕及跨head闭环实际执行。
候选诊断kernel=687.8074us，因额外逐chunk计数，不能直接与只有阶段时钟的基线判定收益。
8条补充分核/初始状态边界：第五轮与候选诊断版均7/8通过；
唯一Case705的FP32 final_state混合容差失败，二者matched_ratio同为0.7554473876953125，
其他所有输出通过。该现象与已有低精度递推问题一致，没有通过修改阈值掩盖。
正常版本现已删除全部时钟和计数探针，仅保留成对发布私有头文件差异，进入正式重编。


## Round11: Case250-only quick screening

Round10 was rejected and round5 restored; see round10/review-summary.json. The user explicitly authorized Case250-first screening. Baseline dtype/key2 compile: 310 seconds. The next candidate fuses direct-Q scaling, preserving both casts; its reviewed plan is round11/qscale-design.md. Results will be recorded in round11/round11-report.md.

Round11 complete as a rejected screening experiment: Case250 CPU golden 1/1; all12 output/None positions bitwise equal. Main kernel +0.15535%, ATK total -0.08297%; no reliable gain. Evidence: round11/round11-reviewed.md.


## Round12 Case250-first experiment

Round12 reviewed and rejected. Original Case250 CPU FP64 golden 1/1 passed; all12 public output/None positions match round5 bitwise. One fresh A/B pair,20 samples each: main kernel679.61685 ->682.19105us (+0.37877%), ATK total766.19275 ->769.01705us (+0.36861%). Stopped after first pair with no gain; no other-case acceptance claimed. All11 kernel source files and both modified installed artifacts match their original backups byte-for-byte; background job ended. Evidence: round12/round12-report.md and review-summary.json.

## Round13: Prepare dependency probe and focused Case250 screening
User requests deeper diagnosis then optimization, with Case250-only quick compilation and minimal accuracy gates.
Reviewed diagnostic design: round13/probe-design.md. It records existing PIPE_S cross-core waits and scalar issue sections, without new sync, in disjoint unused outputScratch records. No change to arithmetic, ownership, queue depths or public ABI.
The original complete installed artifacts and source are restored on exit. Probe overhead is measured before interpretation. The paired diagonal VF candidate was selected and reviewed in round13/pairdiag-design.md, then retained for development after the evidence below.


### Round13 result: paired diagonal VF retained for development
| Model case | Round / status | Evidence | Single candidate | Accuracy | Performance | Bottleneck change | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case250, Ascend950, BF16/FP32/BF16, key2 | Round13 / retained for development | Probe shows AIC score-publication reverse wait and solve-input wait; probe overhead3.62%, so no timing extrapolation | Pair two independent16x16 diagonal solves per32-row direct-score tile; keep identity updates in VF | Public return positions bitwise equal; original CPU FP64 case250 1/1 pass | Four pairs,20 samples each: main684.026975→635.841875us (-7.0443%); ATK770.3425625→722.172425us (-6.2531%) | Uninstrumented PMU AIV Scalar126.749→96.4408us, Vector302.3712→289.492us; AIC MAC/Fixpipe largely unchanged | Source retained. Full generic acceptance remains pending under user Case250-first scope. |

Source exactly matches round13/op_kernel.pairdiag. The currently installed full artifact was restored byte-for-byte because the quick binary only contains one dtype/key2. Next optimization must use round13/pairdiag/bin as its optimized quick baseline, not round11/12 baseline binaries. Review: round13/review-summary.json; report: round13/round13-report.md.

## Round14: Cube merge object reuse, Case250 development screen

Baseline is retained Round13 round13/op_kernel.pairdiag and round13/pairdiag/bin; no reuse of pre-Round13 binary as A. Plan and resource/event review: round14/merge_reuse-design.md. Only existing specialization body and its dispatch call change.

Single dtype/key2 compilation, original12 return-position bitwise comparison, original CPU FP64 ATK Case250 accuracy, then four20-sample paired performance runs (AB,BA,AB,BA) and separate uninstrumented PMU. Main and ATK totals reported separately. User requested this focused development scope; no full wheel or generalized edge-value matrix. Detached runner round14/screen_merge_reuse.py, status round14/screen-status.json, report round14/round14-report.md. Original complete installed artifacts restored at end; source restored too on failure/rejection. Results and final decision are recorded below.

### Round14 result: Cube merge object reuse retained for development

| Scope | Result | Correctness | Performance | Mechanism evidence | Saved state |
| --- | --- | --- | --- | --- | --- |
| Original Case250, Ascend950, BF16/FP32/BF16, key2 | Retain on top of Round13 | 12 return positions bitwise equal; original CPU FP64 1/1 pass | Four pairs,20 samples each: main633.2333125->594.983325us (-6.0404%); ATK719.13355->680.97175us (-5.3066%) | Uninstrumented PMU AIC Scalar285.98085->144.2112us; MAC/AIV largely stable; Fixpipe218.20745->245.3269us, so counters are not additive wall-time components | Source matches round14/op_kernel.merge_reuse; complete installed original restored and byte-compared; no remaining compiler/runner; git diff --check pass |

Quick compile308.8s, bitwise9.5s, CPU accuracy25.0s; no full wheel, all-key build or broad boundary matrix. Detailed report round14/round14-report.md; paired data round14/review-summary.json; PMU round14/pmu-summary.json. This is the user-requested Case250 development screen, not full generic acceptance.

## Round15-22: Case250 performance continuation

Retained implementation: Round14 commit7fa6b338. Current repository process is
AGENTS.md and docs/agents/03-05 V2. The user's latest direction is profiling first,
Case250 quick screening, and wider verification only after stable benefit.
Seven runtime candidates were rejected; two all-register drafts failed the
standalone VF compile before expensive compilation. No candidate code is retained.

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case250 | Round15 / 放弃 | R14 Scalar 调度开销 | 跨 head 复用合并对象 | Original CPU FP64 Case250 pass; bitwise pass | 595.860925->602.741625us (-1.1547%) | 两对均退化 | round15/screen-status.json |
| Case250 | Round16 / 放弃 | 串行小 GEMM 的缓冲管理 | 合并缓冲级数从2改为1 | Original CPU FP64 Case250 pass; bitwise pass | 595.559240->594.909640us (+0.1091%) | 两对退化；充分预热后仅0.109% | round16/warm-status.json |
| Case250 | Round17 / 放弃 | 固定形状仍走通用调度 | 固定16/32小矩阵 Cube 调度 | Original CPU FP64 Case250 pass; bitwise pass | 594.552232->592.996530us (+0.2617%) | 两对退化 | round17/screen-status.json |
| Case250 | Round18 / 放弃 | AIV 对角求解交付路径 | 融合 direct-score Vector 准备 | Original CPU FP64 Case250 pass; bitwise pass | 596.661043->596.832283us (-0.0287%) | 整体无收益 | round18/screen-status.json |
| Case250 | Round20 / 放弃 | R19 当前依赖探针 | 略过求解中的零首行 | Original CPU FP64 Case250 pass; bitwise pass | 592.314562->590.833200us (+0.2501%) | 一对退化，均值仅0.250% | round20/screen-status.json |
| Case250 | Round21 / 放弃 | R19 当前依赖探针 | 相邻目标行配对、共享源行加载 | Original CPU FP64 Case250 pass; bitwise pass | 596.313875->594.835288us (+0.2480%) | 两对退化，均值仅0.248% | round21/screen-status.json |
| Case250 | Round22 / 放弃 | AIC 合并提交与反压区间 | 对角块16→32，Cube合并6→2 | Original CPU FP64 Case250 pass; Akk/attn_out rounding differs | 593.186700->622.551675us (-4.9504%) | 两对均退化，提前停止 | round22/screen-status.json |

Round16 table uses the final four100-sample experiment after10-second warmup,
not its initially noisy20-sample result. Round17/18 also use100 samples.
Round15/22 stopped after two regressing20-sample pairs; Round20/21 completed
four20-sample pairs. All means include every op_summary Task Duration sample.
The reviewed per-candidate resource, numeric and event plans are saved under
roundNN/design.md; the rejected designs are removed from current
Stage documentation. Original values, CPU reference and thresholds are unchanged.

Round19 diagnostic: fresh Round14 baseline590.61069us versus probe598.39767us
(overhead1.31846%). Twenty launches,560 AIC/1120 AIV task12 records. AIC input
wait1.52002us; final AIV diagonal dispatch/backpressure1.53924us. PIPE_S waiting
and asynchronous dispatch/backpressure must be interpreted separately. No new
synchronization, no cross-core sum and no extrapolation to a full-kernel gain.
Scratch and remaining kernel files were checked against the proven probe layout.
Evidence: round19/probe-summary.json and probe_phases/records.json.

Round21 diagnostic PMU: Vector290.5862->290.2348us and Scalar96.50305->95.87735us.
Pipeline times overlap; this does not substantiate a stable end-to-end saving.
Round22 trades larger Vector recurrence for four fewer small Cube GEMMs. CPU
mixed tolerance passes but main kernel regresses4.9504%; the draft is discarded.
Its generic template fallback preservation remains unimplemented in the rejected
artifact, and no broader template compatibility is claimed.

Traceability:
| Design/review item | Candidate location | Evidence/status |
| --- | --- | --- |
| Cube object/buffer/fixed-tile changes | private Arch35 merge functions | rounds15-17 patches and paired screens; rejected |
| Vector preparation fusion | direct-score preparation | round18 candidate.patch, CPU and paired screen; rejected |
| Dependency and scratch lifetime | existing ready/wait and task12 probe | round19 probe-design.md, records and overhead control |
| Zero-row and paired-row VF | ForwardSubDiag16PairStridedRegbase | round20/21 microcompile, CPU, bitwise and paired screens; rejected |
| Diagonal32 and final Cube merge | private direct-score VF/dispatch | round22 design, CPU pass, two regressing pairs; rejected |
| Retained memory/ready-free/slot mapping | unchanged Round14 op_kernel | final source and installed artifact byte comparison |

Final source matches the Round14 snapshot in round22/op_kernel.before.
Complete original installed artifacts were restored byte-for-byte. The installed
package between experiments is older Round5; A/B always explicitly loads Round14
merge_reuse/bin. No untested quick binary is left installed as a full package.

No new full build,48-case matrix, generalized edge matrix, determinism or memory
acceptance was run because no candidate showed stable benefit. The previously
prepared30 small branch baselines have no candidate acceptance result. Formal
generic acceptance remains incomplete; this development validation file stays.
Full result: round22/round15-22-report.md and review-summary.json.


## 第23至25轮最终开发结论（2026-09-21）

继续使用当前 AGENTS.md、阶段03–05 V2以及 CHUNK_DEPENDENT V1
（来源 commit a14961e9dfe8c8264768d67fb447101873da4ecf）。
用户授权继续优化，按先Case250快筛、有稳定收益再扩展验证执行。
云端 codex exec-server，容器4b4a61902120，Ascend950PR_9599，28 AIC/56 AIV，
CANN9.1.0-beta.3。基线显式安装第14轮 merge_reuse/bin；机器默认完整包仍是旧第5轮，
因此默认安装不能充当本轮A/B基线。模型输入、CPU FP64标杆及混合容差保持不变。

| 模型 case | 轮次 / 状态 | 触发证据与唯一主要变量 | 精度结果 | 主核性能/us | 结论与证据 |
| --- | --- | --- | --- | --- | --- |
| Case250 | R23 / 诊断 | FwdH ready/free及3/4head观察区间 | 原CPU通过；12返回位置逐位一致 | 正常594.82267；探针595.75277 | 探针开销0.15637%；AIC state-ready PIPE_S等待2.4439us；round23/probe-summary.json |
| Case250 | R24 / 放弃 | FP32状态原位更新，少一次32KiB UB复制 | CPU和12返回位置逐位通过 | 四组20样本593.0215125→591.178475 | 平均仅0.3108%，一组退化；源码恢复；round24/screen-status.json |
| Case250 | R25 / 开发保留 | AIC L1-ready等待从PIPE_S移到PIPE_MTE1 | 下列已完成范围通过 | 四组20样本596.4306→571.8839625；四组100样本597.8498475→572.1719225 | 全8组正收益；100样本复核降4.2950%；round25/confirm-status.json |

快筛每组20样本，稳定后每组约10秒预热、100样本、AB/BA/AB/BA串行复核。
所有性能值使用无探针 msprof op_summary Task Duration(us)，不扣除样本或筛选最快值。
100样本复核主核加Transpose606.2935425→580.7617675us（降4.2111%）；
ATK全部设备任务683.3275675→657.926105us（降3.7173%，包含IsFinite/ReduceAll）。
PMU显示MAC及AIV Vector基本不变，AIC Scalar活跃时间没有下降；收益与等待位置和调度重叠变化一致，
不能解释为Scalar指令量减少，不能把pipeline计数相加成墙钟时间。

详细诊断构建记录20次×84逻辑核心的实际task/chunk/slot及ready/free提交次数：
每核48或64个chunk、四槽各复用12或16次、跨head计数全部符合预期。
4head FwdH观察区间从约228us降到约205us，前端约361us。
该探针额外开销2.0023%，不计入正式收益；计数只验证提交顺序，不能代替异步引擎完成时间。
源树已撤去探针，原始记录见 round25/probe_phases/records.json、trace-summary.json。
SYS_CNT由launch间隔与op_summary校准为约1000ticks/us，未使用AIC工作频率代换。

已完成验证：

- 原始Case250–257：CPU FP64 mixed_tolerance_bm 8/8通过；没有改动输入分布或阈值。
- Case250：12返回位置逐位一致；100次确定性报告statistic为True。
  统一脚本曾误报未找到报告，实际XLSX已定位并复制到round25/determinism/report，未仅凭退出码判断。
- 扩展分支：21个有限dense分支逐位通过，覆盖4种layout、GVA、奇偶head、batch、初始/最终状态、
  中间量导出和非safe gate。
- 3个新预激活输入夹具1740–1742：由原始模型分布按既有CPU公式激活，固定输入供A/B和CPU共用；
  CPU 3/3、逐位3/3通过，输出有限。保留原错误草稿及失败结果，没有用新夹具替换它们。
- 两个小用例1750/1751：CPU 2/2通过，覆盖8chunk槽复用、同核多head。
  本轮独立CPU精度用例共13/13；扩展逐位分支共24个，另有Case250的12返回位置检查。
- 补充性能两组20样本AB/BA：Case251导出路径689.4409→686.270775us（均值降0.4598%，一组退化）；
  小预激活路径40.987075→41.062175us（增加0.0751us，0.1832%）。两者均不宣称稳定正收益。
  证据 round25/finish-tests-status.json、extra-perf-status.json。

未通过或未完成的范围：

- 初始30分支草稿未全过。tail/true-varlen的1712–1717在第14轮自身重复执行时也不确定；
  差异仅定位到1-token尾块的attn_out（token64或81），其余中间量一致。本轮未修复，不计通过。
- 原草稿1724–1726误将raw正负gate直接用于use_gate_in_kernel=false，
  基线和候选含相同NaN/Inf，不计精度通过。失败输入、A/B输出、重复输出和比较JSON均保留。
- Case250 memcheck300秒、小基线180秒、小候选延长900秒均超时（return124）。
  现场停留在bisheng-tune插桩重编译，无目标ChunkKdaFwd检查报告；不计通过。
  racecheck/synccheck未执行。对应进程组已清理，设备无残留进程。
- 其他dtype/SoC完整构建、全泛化精度矩阵和既有FP32 final-state/尾块问题未完成；
  不将开发结论归档为阶段05正式验收，不删除本文件。

最终仅保留WaitL1SlotReadyMte1的PIPE_MTE1改动和设计/验证记录；Round24及探针未进入源码。
11个kernel文件与round25/op_kernel.candidate逐字节一致，生成目录已同步干净源码；
完整安装.o/.json与round25/installed.before逐字节一致，key1/key2元数据保留。
候选快速产物仅有目标dtype/key2，保留在round25/mte1_ready/bin，不替换完整wheel。
git diff --check通过；设备无进程。实际检查结果见 round25/source-review.json。

复现脚本：bash round25/review_case250.sh，临时安装候选并在退出时恢复完整安装。
完整报告 round25/round23-25-report.md，精简证据包 round25-evidence.tar.gz；
失败证据同样保留，原始大型张量仍在云端round25目录。


## Round26 development experiment (2026-09-24)
Fresh retained Round25 baseline CPU Case250 passed; source byte match verified. Candidate: AIV L1-free default PIPE_S -> PIPE_MTE3; design and dependency/resource review at round26/design.md. CPU and paired performance pending; not adopted.

Round26 completed: CPU/bitwise pass; main570.0811625->569.5323125us,0.0962758%,mixed paired signs; rejected,source/fullinstall restored.

## Round27 candidate (2026-09-24)
Mode4 per-subblock L1 handoff with final FREE drains; reviewed dependencies,resource and counter closure at round27/design.md. AIC/AIV API microcompiles passed. Pending key2 CPU and AB/BA; not adopted.

Round27 rejected: original CPU accuracy failed; baseline repeat bitwise true, candidate repeats nonfinite/nondeterministic. Prepare uses mode4 FREE8/9 and leaves last token; reusing9 as READY admits premature L1 read. No performance measured. Failure artifacts at round27.

## Round28 corrected candidate (2026-09-24)
Same mode4 handoff but isolated flags4/5 and20/21; reviewed inventory excludes Prepare8..11/24..27 and FwdHdirect6/7/22/23. FinalFREE drains retained. Design at round28/design.md. Pending validation,not adopted.


## 第26至28轮开发验证（2026-09-24）

续接原任务，通过SSH隧道调用云端codex exec-server。容器f17331fc82bb，Ascend950PR，28AIC/56AIV，CANN9.1.0-beta.3。源树基于7fa6b338及第25轮未提交优化，显式安装第25轮基线进行A/B；默认完整包仍是旧版本，不参与比较。原始模型Case250、CPU FP64标杆、输入分布、mixed_tolerance_bm不变。

| 候选 | 精度 | 同环境主核/us | 处理 |
| --- | --- | --- | --- |
| R26 FREE等待改PIPE_MTE3 | 原CPU及12返回位置逐位通过 | 四组20样本570.0811625→569.5323125，仅降0.0963%，两组退化 | 撤回 |
| R27 mode4复用9/10 | attn_out逐位和CPU失败，非有限且不确定 | 不进入性能测试 | 与Prepare末尾FREE事件冲突；保留失败证据，撤回 |
| R28 独立mode4事件4/5/20/21及末尾FREE消费 | 已完成范围通过，见下 | 四组100样本571.0068475→564.4298475，降1.1518%，四组均改善 | 保留开发候选 |

R28先四组20样本快筛，再每次约10秒预热、100样本AB/BA/AB/BA复核。全部样本使用无探针msprof op_summary Task Duration(us)。主核加Transpose579.5062725→573.6721425us，降1.0067%；包含IsFinite/ReduceAll的ATK全部设备任务655.49877→650.811875us，降0.7150%，不等同纯算子时间。原始证据：round28/confirm-status.json及CA1–CA4/CB1–CB4。

验证结果：原始Case250–257 CPU 8/8；固定预激活1740–1742 CPU 3/3；小例1750/1751 CPU 2/2，共13/13。原始Case250的12返回位置逐位一致，100次确定性XLSX statistic为True。扩展21有限dense分支及3固定预激活分支逐位一致，输入夹具沿用第25轮。最初预激活CPU测试误用未注册该api_type的executor，三项KeyError并未执行到算子；修正为executor_preactivated.py后通过，原失败报告保留在preactivated_accuracy。

补充性能两组20样本AB/BA：Case251导出路径686.275→674.87345us（降1.6614%），小预激活40.767975→40.4071us（降0.8852%），仅为该覆盖的回归观察。

| 设计追溯 | 实现位置 | 实际证据 |
| --- | --- | --- |
| mode4双AIV READY | WaitL1SlotReadyMte1、SetL1SlotFlagAivToAic | 两个子块分别等待，24分支逐位及13CPU通过 |
| 双AIV FREE和退出闭合 | SetL1SlotFlagAicToAiv、ProcessAiv末尾 | 20次×84核心记录，STATE/VNEW各n次发布及n次消费 |
| 任务所有权及四槽复用 | ProcessAic/ProcessAiv、既有chunk&3 | 实际head映射、48/64chunk、每槽12/16次全部匹配 |
| L1/UB/L0区间与单槽生命周期 | 既有偏移常量及MTE1/FIX依赖 | 地址和容量未变；小用例8chunk复用及同核多head精度通过 |
| 正常性能与探针边界 | 无插桩candidate/probe分别构建 | 正常564.88859、探针571.25601us；探针开销1.1272%，未计收益 |

trace-summary.json和probe_phases/records.json验证实际提交次序与计数，不能把Scalar时间戳当作异步引擎完成时刻。四head FwdH观察区间约197–198us，前端约360us。PMU的MAC/Scalar/Fixpipe活跃时间基本不变、AIC MTE2约229.17→222.61us；不将重叠流水时间相加为墙钟收益或解释成搬运字节减少。

源码已撤去诊断探针，11个kernel文件与op_kernel.candidate逐字节一致；本地隔离工作树手工实现与云端候选去注释/空白后代码一致。快速编译只覆盖目标dtype/key2，产物留在round28/mode4_isolated/bin；每轮实验结束恢复完整安装.o/.json。

正式验收仍未完成：历史1712–1717的1-token尾块基线自身不确定，原1724–1726含错误raw-as-preactivated输入，均未计通过；完整dtype/SoC构建、泛化矩阵和既有FP32 final-state问题未完成。本轮定向memcheck状态另记，不将启动工具或无报告当作通过；保留此开发期validation.md，不归档为阶段05完成。


第28轮收尾：定向memcheck使用--kernel-name=ChunkKdaFwd匹配到了实际key2内核，全block检查，但600秒仍停留在bisheng-tune插桩编译，return124、mss.log为空，不计通过。racecheck/synccheck未执行。超时主进程退出后遗留的本轮进程组已清理，npu-smi确认无运行进程。11个源文件和生成目录均与无探针候选逐字节一致，完整安装.o/.json已逐字节恢复，git diff --check通过。

复现：bash round28/review_case250.sh（约10秒预热、100样本，退出恢复完整安装）。云端精简原始证据包round26-28-evidence.tar.gz包含脚本、设计、失败和成功的报告、op_summary与同步trace；大型张量及二进制不在包内。原始证据导出到本地此前被自动审批拒绝，尚未下载。


## 第29轮：继续优化空间评估（2026-09-28）

通过云端exec-server复连容器bde6535ff823，设备Ascend950PR/CANN9.1.0-beta.3。保留第28轮基线；源码11文件逐字节匹配。原始Case250输入、CPU FP64标杆及mixed_tolerance_bm不变。

新鲜基线CPU250通过，100样本主核564.37609us（sd1.96515us）。唯一候选将Prepare双head、SAFE_GATE BT64/K128/V128的solve-ready默认Scalar等待改为PIPE_MTE2等待，其余发布位置、flag7、计数、数学及资源不变。目标key2编译、12返回位置逐位与原CPU250均通过。

四组20样本562.6661875→559.9028375us，下降0.4911%，一组轻微退化。因结果处在门槛边缘，固定追加一次10秒预热、四组100样本诊断：562.769315→560.42735us，下降0.4162%，四组均改善；主核加Transpose571.1605325→568.9102175us（0.3940%），包含检查算子的ATK设备合计648.880265→646.6242975us（0.3477%）。没有筛样本、重复测到通过或跨日期直接比较。

未达到预设0.5%保留门槛，候选撤回，源码继续保留第28轮。11个源文件和生成目录逐字节恢复，完整安装.o/.json恢复，git diff --check通过，NPU无运行进程。由于没有采用候选，本轮未扩大到13CPU/24分支/100确定性或sanitizer；旧尾块及泛化验收问题仍未解决。

剩余方向：当前每head六次16/32小GEMM的GM中转、workspace行步长及片上驻留值得进一步设计和实测；不能将其它算子的L1双缓冲收益或重叠PMU时间直接换算为本算子收益。跨head对象复用、扩大对角块和value尾部拆分等既有失败方案不原样重试。

原始证据round29：design.md、candidate.patch、baseline-status.json、screen-status.json、confirm-status.json、final-review.json和全部CPU/profiling报告。完整分析见round29-report.md。本轮仅为开发诊断，继续保留validation.md。


## 第30轮：TMP 行距优化（2026-09-28，保留开发候选）

环境：云端exec-server，Ascend950PR、CANN9.1.0-beta.3；第28轮为对照。参考model-agent的Catlass性能skill和本仓Catlass优化指南的512B行距建议；流程和性能口径仍按本仓AGENTS、方案设计V2及CHUNK_DEPENDENT V1。原始Case250的shape、seed、值域、CPU FP64及mixed_tolerance_bm不变。

改动仅在Prepare的六次合并GEMM中将TMP物理行距256→512B，父布局32×128，16KiB plane不变；新增两行与一行注释，替换tensorTmp的layout参数。目标key2编译通过（315秒）。设计/R01–R21评审先于改码；对照原始CPU250、100样本基线562.85955us先于改码。

初筛4×20样本：564.60000→558.726375us，改善1.0403%，四组均改善。正式候选复核4组AB/BA，每组每版本100样本、约10秒预热：主核562.58310→558.15229us，改善0.7876%；主核+1个Transpose 571.11232→567.36460us，改善0.6562%。四组全部改善，达到预设>=0.5%平均门槛。ATK含检查kernel合计650.30855→646.24294us，单独列示，不替代算子耗时。

验证：原始12返回位置逐位一致；13个CPU用例（250–257、1740–1742、1750–1751）全部通过；21个原始分支+3个预激活固定输入与新鲜R28输出逐位一致；Case250 100次确定性通过。未更改任何输入分布、CPU计算或阈值。

实际同核记录：20次×84核心（28AIC+56AIV）全部通过；runtime blockDim28，768个head-pair task/次，四solve slot首次/切换/回绕/尾任务、ready/done计数全部核对；每次4608个TMP写入，实际stride128、最大有效索引3999<4096。记录的是操作提交顺序，不冒充异步完成时间。记录版公开返回逐位一致；其100样本开销约0.7844%，仅用于验证，全部正式性能来自无记录版本。记录源码已撤回。

补充性能不得遗漏：Case1780（原始大shape、disable_recompute=true）2×20短测670.48715→674.11760us，退化0.5415%；为此预先固定追加一次4×100长测，674.16694→673.69682us，变化+0.0697%，两组快两组慢，未确认稳定退化，也不宣称稳定加速。导出路径每调用有2个Transpose；修正最初统计计数假设后复用已采集ECA1，未丢弃/重测该组。Case1781小尺寸预激活2×20为40.41945→40.39543us，变化+0.0594%，视为基本持平。

成对PMU定位：AIC MTE2 225.40640→215.00955us，Fixpipe253.95265→247.27250us；MAC106.34035→106.34935us、AIV Vector290.07495→290.55150us。只支持搬运路径改善假设，不将重叠管线时间相加。

收尾：11个源文件和生成文件逐字节匹配无记录候选；原完整安装.o/.json逐字节恢复，git diff --check通过，NPU无运行进程。云端源码与本地隔离工作区保留候选，默认完整OPP安装不被单key临时产物替代。复测命令：bash round30/review_case250.sh，退出自动恢复完整安装。

原始证据round30：design.md、candidate.patch、baseline-status.json、screen-status.json、confirm-status.json、wider-status.json、trace-summary.json、trace-run-status.json、extra-perf-status.json、export-confirm-status.json、final-review.json及CSV/XLSX/原始trace。完整说明round30-report.md。只作为本轮开发结论保留validation.md；旧尾块/FP32 final_state/完整dtype与SoC覆盖及sanitizer正式验收仍未完成。本轮没有重跑此前600秒卡在插桩编译的sanitizer，不计为通过。NZ临时布局和直接L1驻留仍是未实测后续候选。

## R31 开发验收记录

完整 R30 wheel 已全量构建并安装，366 个 OPP 文件与 wheel 逐字节一致。基线 smoke：
2000–2002 attn_out NaN；2003–2008 final_state 精度失败；2009–2011 全输出通过；
2012–2014 w/final_state 失败；2015 K=V=256 超时停止，2016–2017 未执行。
证据 acceptance-r31/baseline-smoke、diagnostic-tail、ct-tail。
R31 通用 PostWU scratch 和尾块 QH Cube 修复按 design.md 实施，当前状态：待编译、待实测。

## R31 值域校准补充
CPU 校准36组：6种 smoke 几何 × 原scale/1 × v_scale=0.05/0.2/1；
同一份量化输入的 FP32 正确实现均通过 FP64 golden 的默认混合容差。
原 BF16 小振幅 attn/Aqk 的零值及2倍错误可能通过阈值，不能以原 smoke 宣称值域充分。
新 dimensions/branches 用例采用 scale=1、v_scale=0.2（显式scale边界保持原值），
无阈值变更；既有 model 和失败 smoke 的原值域保持，用于不变条件回归。
全部详细指标留在 calibration/results.json；正式验收前还检查新用例的代表几何与三个种子。

新增域156组 CPU 校准均通过。预激活分支再比较 data_scale=0.08/0.12/0.2/0.3，
两种 dtype、三个种子共24组正确实现均通过；选择能检出 attn 全零/2倍错误的最小档0.12。
CT Tool 0.9.1 图已检查：CPU FP32 正确输出贴合 y=x、误差约1e-7、无非有限值；
全零错误明显离开对角线，误差扩散到全部token。图为CPU值域校准，不是NPU精度截图。

用户于2026-09-29确认：保持指定参考代码的中间输出 dtype 接口，并在交付报告注明与任务书第2.2节 FP32 表述的差异。

R31 原 host 单 chunk 路径验证：{"tests": [{"id": 2012, "exit_code": 0, "pass": true}, {"id": 2015, "exit_code": 0, "pass": true}], "revert_host_split": true}
