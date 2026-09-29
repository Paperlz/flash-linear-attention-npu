# ChunkKdaFwd 设计

方案设计规则版本：V2。当前开发候选为第14轮基线上保留第25轮 MTE1等待、第28轮独立mode4 L1交接及第30轮TMP的512B行距。
第15至22轮、第24轮候选未采用；第23轮为诊断。泛化正式验收未完成。
性能与验证边界见开发期 [validation.md](validation.md)。

## 目标

1. 顶层接口对齐非 CP 的 FLA `chunk_kda_fwd`。
2. 不新增公开算子原型；A5 快路径复用既有 `ChunkKdaFwd` 原型和外层 kernel 入口。
3. A2/A3/A5 使用同一数学定义；A5 保留 regbase 双发射特化。
4. 输入 layout 与输出 layout 解耦。
5. FwdH 同时服务 KDA 与 GDN，并支持可选 scalar gate、key-wise gate 和 `state_v_first`。

## L2 调度

```text
raw g -> ChunkKdaFwd[
    gate cumsum -> Prepare/Post-WU -> FwdH -> Finalize
] -> attn_out
```

`aclnnChunkKdaFwd` 做公开 layout 的连续化和必要视图转换。A5 的 BF16、chunk=64、K=V=128
dense 对齐快路径保持单次物理 `ChunkKdaFwd` L0；A5 其他多 chunk 场景将同一个私有 L0 按
Gate/Prepare、Post-WU、FwdH、Finalize 四个阶段依次提交，使阶段间通过物理 launch 边界重置事件状态。
A2/A3 和单 chunk 场景仍使用单次物理 L0。阶段选择仅使用私有 `stage` 属性，不增加公开属性、
接口字段或独立算子原型。

## 阶段职责

### KdaGateCumsum

将 raw/已激活 gate 转为 FP32 chunk-local log2 累计值：

```text
gk = cumsum(gate) / ln(2)
```

该算子同时保留独立 L2 接口供 GDN2 调用，输入和输出固定为 BNSD/NTD。

### Prepare

只读取 `q/k/v/gk/beta` 及变长元数据，产生：

```text
Aqk, Akk, qg, qg_scaled, w_seed, u_seed
```

矩阵计算和三角求逆使用 FP32 累积；公开中间量在写回时转为 q dtype。

### Post-WU

只读取 `k/gk/w_seed/Akk/u_seed`，产生：

```text
w, u, kg, v_new_seed
```

`Akk` 的 head 循环按 `H_v` 执行，GQA 映射只在读取 q/k head 时换算，避免按 `H_k` 重复或漏算。

### FwdH state propagation

读取 `kg/w/u/gk` 和可选 `initial_state`，计算 chunk 间递推：

```text
v_new = u - w @ h_prev
h_next = exp2(gk_last) * h_prev + kg^T @ v_new
```

arch35 路径复用与 `ChunkGatedDeltaRuleFwdH` 相同的数学实现；其他场景在 `ChunkKdaFwd` 内嵌
共享 FwdH 实现。独立 GDN L0 原型继续保留给其他调用方，key-wise `gk` 固定使用 `exp2`。

### Finalize

只读取 `qg_scaled/Aqk/v_new/h`，计算：

```text
attn_out = qg_scaled @ h + Aqk @ v_new
```

kernel 内直接按 BSND/TND 写出 `attn_out`。供反向使用的中间量保持 BNSD/NTD。

## 状态布局

内部递推统一使用 `[...,K,V]`。`state_v_first=true` 时，L2 在进入 FwdH 前转置 initial state。
内部 `hCompute` 始终保持 head-major 供 Finalize 消费；公开 `hOut` 在 L2 导出边界转为
sequence-major，并按 `state_v_first` 决定末两维顺序。`final_state` 按序列排列，与 FLA 顶层
输出一致。

## 重计算策略

L2 不理解 autograd 重计算策略。`final_state/gk/w/u/qg/kg/v_new/h` 是相互独立的
`OPTIONAL_OUTPUT`；非空指针表示导出，空指针表示不公开该结果。单 launch 路径为隐藏输出传递
固定 ABI 占位，并由 tiling 在 kernel workspace 中承接实际中间结果。A5 四段 launch 路径将
阶段间依赖的 `gk/w/u/qg/kg/v_new/h/final_state` 和私有 `qg_scaled/u_seed` 物化为 executor 内部张量，
使后续 launch 不依赖前一 launch 的 kernel workspace。公开输出存在时直接作为内部目标使用。

Python/legacy 包装层对齐 fla-org `chunk_kda_fwd` 提交
`0f0f0c97af39343855b43bbbaddcedfda5cb9d77`：

- `Aqk/Akk` 始终返回。
- `disable_recompute=false` 时不保留 `w/u/qg/kg/v_new`。
- `disable_recompute=true` 或 `return_intermediate_states=true` 时保留公开 `hOut`。
- `use_gate_in_kernel=false` 或 `disable_recompute=true` 时保留 `gk`。
- `final_state` 只在 `output_final_state=true` 时创建公开输出。

内部 `hCompute` 与公开 `hOut` 是两个生命周期：`hCompute` 是 FwdH 到 Finalize 的必需
head-major 阶段结果；`hOut` 为空时，单 launch 路径由 kernel workspace 承接，四段路径由 executor
内部张量承接。`hOut` 非空时，L2 提供 head-major 临时输出并在导出边界转为 sequence-major。
该规则对齐非 CP 的低层 12 返回值接口；
第 12 项 `initial_state` 由 Python 层原对象透传。

## 模板化方案与 tiling key

`ChunkKdaFwd` 只有一个外层 `op_kernel/chunk_kda_fwd.cpp` 入口和一个私有 L0 类型。A5 实现位于
`op_kernel/arch35/*.h`，host 侧 A5 模板选择位于 `op_host/arch35/*.h`。Prepare、Post-WU、
Finalize 的内部实现头与统一 kernel 入口同属 `chunk_kda_fwd/op_kernel/`，不存在对应的独立 L0
原型或 `.cpp` 入口。A5 四段路径只是用不同私有 `stage` 属性连续调用该入口。

- `tiling key=1`：非 chunk=64、K=V=128 场景的通用模板族。
- `tiling key=2`：chunk=64、K=V=128 模板族，包括 dense、tail 和 varlen。

两个 key 是同一 L0 的编译期场景变体，不是平台编号、独立算子或独立接口。A2/A3/A5
均生成两个 key；同一个 key 内再由编译架构选择根目录通用实现或 `arch35/` 实现。host 的
`SetTilingKey` 只检查 chunk、K、V，不检查 SoC。

在 arch35 上，key2 的 dense 对齐场景使用单 launch 融合流水和 arch35 FwdH。A5 多 chunk 的
tail/varlen 以及 key1 泛化场景使用四段 launch，并在 FwdH 阶段复用共享实现。其他架构在同一
key2 下使用其对应单 launch 实现。tiling key 和私有 `stage` 均不改变公开算子原型、输出契约
或数学定义。

## 性能设计

- Prepare 的右矩阵在 L1 驻留，避免 K/K^T 重复搬运和重复转置。
- AIC 使用 L1/L0 双缓冲组织 MTE2、MTE1、Cube、Fixpipe。
- AIV 使用输入 staging ping-pong，使下一 tile MTE2 与当前 tile VEC 重叠。
- A5 VEC 路径使用 regbase 双发射；数值主计算仍保持 FP32。
- inter-sub-chunk 合并使用独立 workspace 区域，避免阻塞主 tile 流水。

性能结论统一使用 msprof 的 op_summary 中 Task Duration(us)。主核、主核加 Transpose、
ATK 全部设备任务分别统计；pipeline 活跃时间仅作瓶颈诊断。目标回归 case 定义在
`tests/atk/chunk_kda_fwd/atk_chunk_kda_fwd.json`。

## 验证矩阵

- 平台：A2/A3/A5。
- dtype：FP16/BF16。
- layout：BSND/BNSD/TND/NTD。
- gate：raw/已激活、safe true/false。
- Shape：K=128，V=128/256，chunk=64/128，dense/varlen/tail/GQA。
- 属性：final state、重计算策略、`state_v_first`。

## Prepare 中 K 与 score 缓冲区的生命周期

A5 safe-gate 的分块 Prepare 在同一 UB 区域保存原始 K 和缩放后的 score W。
`kg = k * exp2(gk_last - gk)` 必须使用原始 K；不能在 score W 写回后再从
该区域读取 K，否则会重复引入 beta 与 score gate 缩放。

融合向量函数在原始 K 仍位于寄存器时计算最终 kg，并在读取 V、生成 direct V
后才写入复用 V 的暂存区。该调整保持现有 L0、公开接口、shape 支持范围、
BF16 舍入方式和 workspace 布局，使用原有寄存器和写回地址。

回归以 ATK CPU FP64 golden 检查完整输出，重点覆盖导出路径和 chunk 最后一行
`kg == k` 的数学恒等式；同时复测 Case 250 性能及确定性。


## Round11 experiment: Prepare Q scaling fusion

The independent Case250 screening candidate is specified in round11/qscale-design.md. Only the existing no-qg-export fused Prepare path moves the two-rounding Q scale into its VF; all memory intervals and cross-core protocols are unchanged. Candidate status: awaiting device results.

Round11 review: Q scaling fusion rejected. Case250 kernel 683.1805 -> 684.2418 us; source and original installed artifact restored. The prior experiment is not an adopted design change.


## Round12 experiment: invariant reference preload

Reviewed candidate: round12/reference-design.md. Hoist both read-only gate-reference loads across VF rows; preserve separate midpoint and final exponent arithmetic, all memory intervals and sync. Rejected after Case250 screening: main kernel 679.61685 -> 682.19105 us (+0.37877%); the candidate is not adopted and the original source was restored. See round12/round12-report.md.

## Round13 candidate: paired direct-score diagonal solve
Reviewed design and precise memory/synchronization review: round13/pairdiag-design.md.
The direct-score32-row AIV stage pairs its two independent16x16 diagonal solves into one VF. Row arithmetic order, FP32 precision, input/output ownership and cross-core protocol remain unchanged; diagonal identity stores use an explicit diagonal mask. Retained as the development implementation after Case250 screening: main kernel684.026975 ->635.841875us (-7.0443%); ATK device770.3425625 ->722.172425us (-6.2531%), four pairs. Case250 bitwise and CPU FP64 checks pass. Full generic acceptance remains pending; no other cases were run in this screening.

## Round14 candidate: reuse Cube merge scheduling state

Development scope is original Case250 on Ascend950 (SAFE_GATE, BT64, K128, V128), extending retained Round13. Detailed reviewed delta: round14/merge_reuse-design.md.

The actual old dispatch uses six calls to CubeGemmSolveSub, each constructing and destroying the same Catlass BlockMmad. Candidate ComputeAkkMergeCubeWorkspaceArch35 executes the same two16x16 merge chains and one32x32 merge chain with one BlockMmad instance. The pre-existing unused hand-written method is replaced. Original IEEE FP32 GEMM operations, GM row-major data and six FIX_MTE2 completion waits remain. No extra workspace, changed cross-core lifecycle, arithmetic change or claimed new overlap.

Resource allocation remains L1 A/B two stages each (64KiB total), L0A two stages (32KiB), L0B two stages (32KiB), L0C one stage (16KiB), all with 64x64 max tiles. Existing Catlass stage free events remain balanced while indices cycle across six calls; destructor drains at the end. Eight initialization SetFlag and eight destructor WaitFlag calls execute once instead of six times per head. No other operation interleaves inside this group; L1 resident caching is disabled. Direct FP32 L0C-to-L1 residency is deferred because current L1 C0=8 layout differs from Fixpipe NZ C0=16.

This candidate is retained for Case250 development after four paired A/B improvements and original CPU FP64 accuracy pass. Full generic acceptance remains pending. Performance and evidence are recorded in validation.md.


## 第25轮：AIC L1-ready 在 MTE1 等待

第25轮历史实现（事件协议由下述第28轮替代）：WaitL1SlotReadyMte1 调用 CrossCoreWaitFlag<0, PIPE_MTE1>(readyFlag)。
仅改变等待的流水，mode0、flag9/10、发布方、调用顺序、计算、舍入及资源均保持不变。
第23轮测得 AIC Scalar 在 state/v_new ready 等待约2.4439/0.9364us；
让实际消费者 MTE1 等待，Scalar 可以提前下发后续工作，L1 读取仍受同流水等待保护。
AIV 的 FREE 等待仍使用 PIPE_S。公开接口、ABI、模板选择和支持范围不变。

Case250 四组20样本快筛和四组100样本复核全部正收益。复核主核
597.8498475→572.1719225us，降低4.2950%；主核加Transpose降低4.2111%。
正式性能不含探针。CPU精度、逐位比较、确定性及同步提交记录的结果和限制见 validation.md。
完整评审原件：round25/design.md；这里只收录保留实现的依赖与资源结论。

### 依赖与同步评审
Fused path each chunk:
AIC ComputePostWU submits W/U Cube and direct Fixpipe publication first.
AIV publishes current H to L1, consumes W into L1 operand W, then STATE_READY on MTE3.
AIC MTE1 waits STATE_READY before copying W and H from L1. Existing MTE1_M event prevents Cube consuming either L0 operand before its load.
AIC produces WH then QH; STATE_FREE remains on FIX after WH publication, when H is already resident in L0B and is reused there for QH.
AIV produces vnew, casts/stores it in L1, then VNEW_READY on MTE3.
AIC MTE1 waits VNEW_READY before Kg/Vnew/Aqk operand loads; existing MTE1_M prevents Cube consuming incomplete L0. VNEW_FREE remains on FIX after update publication; L0B retains Vnew for output.
Moving the waits to MTE1 lets Scalar dispatch independent work ahead; it cannot make any L1 load pass its readiness wait.
Nonfused path: AIV publishes STATE_READY after H copy. MTE1 first waits STATE_READY, then existing MTE2_MTE1, so both independent GM inputs and recurrent H are ready before their loads.
Prefetch ring depth4 remains protected by existing MTE1_MTE2 flags. MTE2 prefetch cannot overwrite a slot until its previous MTE1 consumers release it. L0 M_MTE1 and L0C FIX_M waits remain unchanged. Direct mode4 FIX/V flags unchanged.
No new ready/free counts, no producer moved, no first/last task branch or drain changed. The final unconsumed L1 FREE notifications are pre-existing; kernel end/drain behavior unchanged. Every live L1 read is strictly downstream of its same-pipe wait.

### 任务与资源
All math, dtype, reduction order, rounding and masks unchanged. Case2501536 head-chunks,28AIC/56AIV,96headtasks =>12Cube/24Vector process4heads,16Cube/32Vector process3heads; eachhead16chunks andslot=chunk&3 wraps4times.
UB byte map (all unchanged): state[0,32768)FP32[64,128]; stateTyped[32768,49152)BF16[64,128]; direct[49152,81920)FP32[64,128]; out1[81920,98304)FP32[32,128]; vnew[98304,114688)FP32[32,128]; ioTyped[114688,131072)allocation16KiB active8KiB BF16[32,128]; gate[131072,131328)FP32[64]; free[131328,253952). Initial/recurrence/export lifetimes unchanged.
L1 byte map: W0[0,16384),Q0[16384,32768),H[32768,65536),Kg0[65536,81920),Aqk0[81920,98304),Vnew[98304,114688),slot1W/Q/Kg/Aqk[114688,180224),slot2[180224,245760),slot3[245760,311296),Akk slots4x8192[311296,344064),U slots4x16384[344064,409600); free[409600,524288). W/Q/Kg tiles BF16[64,128]; Aqk/Akk logical BF16[64,64] (Aqk occupies reserved16KiB), U BF16[64,128], H BF16[128,128], Vnew BF16[64,128]. H/Vnew single slot owned by existing STATE/VNEW ready/free; independent ring reuse by MTE1_MTE2.
L0A state[0,16384),vnew[16384,32768),post[32768,40960),remainingfree to65536. L0B state[0,32768),vnew/post time-shared[32768,65536). L0C single region max[0,65536)FP32[128,128] protected by existing FIX_M. No added GM/workspace, data-copy bytes or resource reservations.
R01-R21 delta review passes: changing the wait pipeline narrows synchronization to actual consumer; all producer/consumer dependencies, memory capacity, formula, ownership and lifetimes preserved.


## 第28轮：L1交接使用独立mode4事件

当前保留第28轮开发候选；第25轮的MTE1消费者等待继续保留，以下同步协议替代其mode2发布/mode0等待和未消费的末尾FREE。公开接口、dtype、公式、舍入顺序、所有数据地址及上面的资源表保持不变。

唯一主要变量是L1交接事件协议。AIV通过PIPE_MTE3发布本地STATE_READY4或VNEW_READY5；AIC在PIPE_MTE1分别等待4/20或5/21两个子块，之后才读取对应L1。AIC完成既有WH/update的FIX发布后，通过PIPE_FIX向两个子块分别发送FREE4/20或5/21。AIV覆盖单槽前仍在PIPE_S等待本地FREE，不改变发布位置或数据消费者。

mode4事件范围：L1使用4/5/20/21；既有direct UB使用6/7/22/23；Prepare使用8–11/24–27。Prepare结束后FREE8/9仍可能有剩余通知，因此绝不能按“数据完成”判断这些事件可复用。第27轮复用9号导致非有限输出和不确定性，已撤回。mode2 Matmul事件与本地HardEvent各属独立资源域。

对每个实际参与的AIV、每类L1数据，n次发布对应首轮跳过FREE、n-1次覆盖前等待和退出前1次FREE消费；无工作时不等待。AIC等待两份READY并发布两份FREE，跨head保持已有计数。计数深度至多1；最终等待位于最后一次direct-free之后，不形成环。

H在WH前已载入L0B，QH继续复用L0B；Vnew同理在update后仍驻留L0B供输出使用，因此原FREE发布点覆盖全部L1消费者。既有MTE1_M、M_MTE1、V_MTE3、MTE3_V、四槽prefetch释放和direct UB同步均保留。L1峰值409600B、UB131328B、L0A40960B、L0B65536B、L0C峰值65536B，新增数据空间及搬运量均为0。

R01–R21差异评审沿用上面的完整任务/资源图，复核两个AIV半块的依赖、单槽覆盖、跨head和最终计数闭合。云端实际20次×84逻辑核心记录验证task、chunk&3、所有READY/FREE提交计数；这不是异步流水完成时间证明。探针从最终源码撤回。

同环境四组100样本主核571.0068475→564.4298475us，全部改善，降低1.1518%。13CPU精度、24有限分支逐位及Case250 100次确定性通过。完整构建与泛化验收未完成，限制和内存检查结论见validation.md。设计原件及原始证据留在round28。


## 第30轮开发候选：合并 TMP 的 512B 行距

2026-09-28，在第28轮上仅调整 ComputeAkkMergeCubeWorkspaceArch35 的 tensorTmp 布局，仍适用于现有 Ascend950 SAFE_GATE、BT64/K128/V128 分派。六次 IEEE FP32 GEMM 的公式、顺序、输入和有效 tile 不变；每两次仍为 TMP = D_lower @ X_off，再 X_off = TMP @ D_upper。

TMP 父布局由 RowMajor(64,64) 改为 RowMajor(32,128)，物理行距256→512B，有效块仍16×16或32×32。TMP(i,j)=tmpBase+i*128+j；最大有效索引31*128+31=3999，小于原4096元素plane。父布局占用仍16384B，原28core×4slot×5plane工作区9175040B、基址和plane/slot步长不变。X仍使用64元素行距，邻接RAW/IDENTITY别名和所有UB/L1/L0容量不变。

Fixpipe写TMP与MTE2读取均从tensor stride取得128元素行距，m/n仍为half；padding不作有效操作数。原FIX_MTE2 event7、ready7/done6与四slot生命周期全部保持。没有新增转存次数、同步或数值变换。R01–R21差异评审与逐项地址证明存于round30/design.md；继承既有映射及GM中转，未声称完成全部V2映射迁移或L1驻留。

这是通过本轮门槛的开发候选，正式泛化验收仍未完成。精度、原始目标性能、补充导出场景短测与长测的不同结果、同核执行记录及限制均记录于validation.md。正式代码中不含trace。

## R31 合法范围修复设计（待验收）

本轮不修改公开接口、CPU 标杆、随机值域或精度阈值。首批 18 个三种子烟雾用例复现了尾块 NaN、
通用 PostWU 的 W 原地覆盖、FP32 final_state 精度失败和 K=V=256 卡住。以下为实现前评审方案。

- PostWU：64×128×128 完整块的已优化路径保持其现有分工；其他块由唯一 producer 生成 W。
  curT≥16 时 AIC 按 N≤128 分片执行 Akk @ wSeed，FP32 结果写入已有 postWuScratch。
  scratch 索引为 ((b*HV+hv)*T+start+row)*K+col，总量 B*HV*T*K*4 字节，
  不使用带 NT*BT padding 的索引。AIC 完成 W/U 后通过既有 syncDoneFlag 通知两个 AIV，
  AIV 再 cast W 并计算 kg。输入 wSeed 在所有 Cube 读取结束前不覆盖。
- curT<16：沿用短归约 Vector 边界处理，W 先写相同 FP32 scratch，两个 AIV 在 MTE3 屏障后
  才复制到 w；AIC 不计算同一 W/U。AIV 拆行及空行分支均参加屏障，避免部分参与者死锁。
- Finalize：QG @ H 的归约维 K 始终≥16，尾块也使用 Cube；仅 Aqk @ VNew 的短归约使用
  Vector。AIC 发布完成标志后 AIV 读 QH，消除短尾 QH 的逐元素标量路径。
- 资源：上述 scratch 已由 host 分配，生命周期为 PostWU 开始至复制结束，不新增 workspace；
  Cube N 片≤128，L0C≤128*128*4=65536 字节。Vector 临时量复用原缓冲区。
  复用既有 syncDoneFlag，无新增跨核 flag。典型块不参加该通用通知。
- 验证：先复跑 18 个 smoke 的全部 12 个输出、3 个种子、NaN 内存污染；
  再验收维度组合、tail、布局、GVA、变长、可选输出及负例。每个 NPU 诊断有 60 秒退出边界。
  FP32 final_state 的精度修复单独设计和验收；当前不得标记为可交付。

### R31 FP32 最终状态修复方案（待验收）

可选最终状态使用 token 递推直接保留 FP32 状态，避免已舍入的 w/u/kg/vNew 反馈到 FP32 输出。
独立 CPU 验证：六类 smoke 的 FP64 递推与现有 FP64 chunk 标杆最大差≤2.5e-16，
FP32 递推最大差≤4.1e-8。公式逐 token 为 h_decay=h*exp(g_t)，
dot_v=sum_k(k_t*h_decay)，delta_v=beta_t*(v_t-dot_v)，h=h_decay+k_t*delta_v。
这里的单 token 点积归约和秩一更新采用 Vector；已有 chunk GEMM 的 Cube 路径不变。
gate 使用已计算的 FP32 base-2 chunk 累加量相邻差，chunk 首 token 的前值为零；
相邻差转自然指数后求 exp。测试同时覆盖 raw gate 和预激活 gate 以核验累计差误差。

Stage：新增可选 Vector 状态输出阶段，仅在 FULL 或 FINALIZE 的已有阶段完全结束后执行。
Nwork=seqNum*HV，CG=1，coreIdx grid-stride，两个 AIV 沿 V 各处理 V/2；
共享同一 q/k source head 的 hv 使用 hk=hv/(HV/H)。变长零长度序列复制初始状态或零。
所有输入均按 host 转换后的真实 stride 读取；初始和最终状态使用内部 [N,HV,K,V]。

每 AIV UB：state [0,131072) FP32 K×align64(V/2)；
prev_gate [131072,132096)，current_gate [132096,133120)，decay [133120,134144)，
key [134144,135168)，value [135168,135680)，beta [135680,135936)，
typed 输入 [135936,136768)。分配上界 144KiB，不使用 L1/L0/额外 GM workspace。
state 在每个序列整个 token 循环驻留；每 token 一次 VF 执行点积和更新，
VF 第一遍只读原状态求 dot，第二遍写状态，避免 VF 内 UB store/load 依赖。
输入 MTE2_V ready、VF 完成 V_MTE2 free、最终写回 V_MTE3/MTE3_V 闭环。
跨阶段 SyncAll 仅在全部 AIC/AIV 已退出前阶段后发生；其后无跨核数据共享。

精度和性能风险：额外读取 k/v/gk/beta，仅请求最终状态时执行；Case250 无最终状态，
保持原主性能路径。全量测试中单独报告含最终状态的耗时，不将未测试场景记作通过。

### R31 单 chunk 的事件生命周期验证

原 host 单 chunk 路径在 PostWU/FwdH 真正问题修复后，T=17/CS128 的最小和最大维度均通过。
因此撤回临时强制拆成四次 launch 的候选；host 维持原有选择条件，没有额外发射开销。
### R31 K>128 状态 Cube 容量修复

CANN 转储确认最大维度卡在 FwdH。状态更新 GEMM 的 M=K、N=V；
scheduler 的 vBlockSize=vHeadDim，没有外层 V 分片。撤回此前 V≤128 分片的错误假设。
K>128 的 KDA 私有模板改为 L1=(256,256,64)、L0=(256,256,32)。
L0A/B 各双 bank 32768B，L0C 单 bank 262144B；
GEMM L1 双 bank 共131072B，VUpdate 预留至多131072B，总量≤512KiB。

复用的 GDNFwdHKernel 根据 L0C tile 字节数选择能容纳的 1 或 2 个 bank。
已有 128×128 和 128×256 模板仍为 2 bank，只有 KDA 新 256×256 模板为 1 bank。
接口、数学语义、任务调度、GM/UB 所有权和 ready/free 协议不变；
单 bank 按现有 FIX_M 事件完成写回后复用。模板自身静态容量断言必须通过。
共享头的引用者为 chunk_kda_fwd 和 chunk_gated_delta_rule_fwd_h；
后者需构建并回归原有 128×128/128×256 场景。融合 GDN 使用自己的内部头，不受此次修改影响。
最大维度、已有 smoke 和全部已规划维度组合均待实测。

### R31 短归约首次异常阶段

CANN dump 已确认 T=65 最后 token 的 qgScaled、Aqk、vNew、h 全部有限，NaN 首次出现在
FINALIZE。将 curT<16 的 Aqk/VNew 短归约改为一次批量搬入 VNew 后用 VF 寄存器累加，
不再通过 GetValue 在 Scalar/Vector 之间逐系数切换。
UB 复用 vecBuf：[0,24576) 最坏 8×256×3 个 FP32 输出暂存；
[32768,48128) 最坏 15×256 个 FP32 VNew；[65536,73216) 对应 BF16/FP16 输入。
系数位于 gateWritebackBuf 的 typed[0,30) 与 float[1024,1084)，均不重叠。
同一个 AIV 的两个暂存区域在同一 VF 完成后释放，不新增 GM 或同步 flag。待污染内存用例实测。

### R31 Fix3 诊断结论

Fix2 的尾块 attn_out 与最小维度 W 已有限且通过单种子诊断。
最终状态的 BSND 误差来自 beta 被 host 单独转为 head-major；
beta 地址必须始终使用 (b*HV+hv)*T+token，k/v 保持 tiling 指定布局。
变长 CANN dump 显示 PostWU 只有输入没有输出：当每核 task 数小于 pipeline 深度时，
AIC 原分支仅处理完整块，AIV 却等待 17/63 行尾块结果。补充同一通用尾块调用；
短于16行仍由两个 AIV 处理，mode1 子块屏障仅同步该 core 的两个 AIV。
不增加测试范围，使用原计划的 18 个 smoke 先验收。

### R31 Fix4 定向实验：通用 score Cube 补齐 N

case5108（T64/K16/V256，key1）CANN 只有 PREPARE 输入、无输出；强制拆阶段仍超时，故撤回 host 实验。
与已通过的尾块相比，完整块 score 的 N 从16逐步增至64；尾块一直以 BT 为 N。
候选将通用 score Cube 的 N 固定为 BT，沿用已有 ZeroScoreScratchPadding 对未生成列置零。
有效下三角公式、K、输出行距及有效范围不变；上三角最终仍由现有 solve mask 排除。
最大 tile 仍为既有 64x128，UB/L1/L0/workspace、ready/free 次数均不变；目标快路径的专用 score 不受影响。
此为待实测假设：先构建 BF16/gFP32/betaBF16 的 key1，复现5108；失败则撤回，不宣布修复。

### R31 Fix5：限制 direct-score 专用路径

Fix4 列宽候选在 case5108 仍60秒超时，已撤回。
已确认分派不一致：AIC 的双 head direct-score 条件只有 SAFE_GATE、curT64、rowCount32；
AIV 的 UB 初始化和 direct-score 消费却只为 COMPILE_BT64/K128/V128 编译。
因此 key1 的 T64/K16/V256 进入 AIC direct-score，AIV 不发布对应 FREE，形成死锁；
AIV 的 useDirectScoreUb 也错误跳过通用 solve 输入准备。
修复将这两个 direct-score 条件统一限制为上述既有编译形状。其他形状仍用现有 GM score/solve 路径。
不改变双 head 所有权、缓冲容量、工作区布局、数学结果或 ready/free 次数。目标 key2 性能路径完全保留。
R01–R21 差异评审：把 producer/consumer 的选择条件对齐，不新增数据流或同步。
先验证原5108–5110与18 smoke，再继续既定矩阵；禁止把已超时的 Fix4 算作通过。
