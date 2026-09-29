# chunk_kda_fwd 第二轮优化：gk 精简写回

## 优化目标

Prepare 阶段的 `MaterializeRawGateChunkArch35` 在每个 chunk 把整段累积 gate
写回 GM 的 `gk`，但当 FwdH 已融合 Post-WU 且用户不导出 `gk` 时，下游唯一
消费者是 FwdH 的 K-gate 状态更新，它只读取 chunk 的**最后一行** gate。
其余行属于纯写放大。

## 代码修改

修改文件：

`fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h`

- 新增 `fusePostWuIntoFwdH_` / `storeGk_` 两个 tiling 标志与
  `GateScratchMaterializeLastRowOnly()` 判定。
- 仅在 `computeGateInPrepare && fusePostWuIntoFwdH && !storeGk` 时生效。
- UB 内完整 gate 累积保持不变；GM 写回由「整个 chunk（`currentRows * K_`）」
  改为「chunk 最后一行（`K_`）」。
- 工作区布局、地址偏移、其他分支写回逻辑均未改动。

## 写回量核算（Case 250）

| 项 | 修改前 | 修改后 |
| --- | ---: | ---: |
| 每 chunk 写回 | 64 × 128 × 4 B = 32 KiB | 1 × 128 × 4 B = 512 B |
| 每次调用写回（16 chunk × 96 head） | 48 MiB | 0.75 MiB |

## 同口径 A/B（ATK `performance_device`，交替执行）

基线 = 第一轮优化版本（`e49673dc`），A/B 使用同一条 ATK 命令、同一组输入，
每次重新安装对应 wheel 后交替测量 3 轮。

| 轮次 | 基线 (us) | 精简写回 (us) |
| --- | ---: | ---: |
| 1 | 784.13745 | 785.64575 |
| 2 | 784.84180 | 786.20585 |
| 3 | 784.89380 | 785.93450 |
| 均值 | 784.624 | 785.929 |

- 差值 **+1.30 us（+0.17%）**，即端侧耗时没有下降。
- 该差异远小于 ATK 波动率（任务书标杆波动率 2.52%），不构成实际收益。

## Profiler 证据（同一 kernel 的 AIV 流水）

`kernel_details.csv`（20 次采样的 MIX_AIC 行）：

| 指标 | 基线 | 精简写回 | 变化 |
| --- | ---: | ---: | ---: |
| `aiv_mte3_time` (us) | 152.04 | 136.07 | **-15.97（-10.5%）** |
| `aiv_mte2_time` (us) | 115.94 | 105.63 | -10.31 |
| `aiv_vec_time` (us) | 323.07 | 322.16 | -0.91 |
| `aiv_scalar_time` (us) | 135.49 | 135.80 | +0.31 |
| `aic_scalar_time` (us) | 285.97 | 285.30 | -0.67 |
| kernel `Duration` (us) | 701.37 | 702.78 | +1.41 |

- 写回确实被消除：16 us ≈ 48 MiB / 3.1 TB/s，与预期写回量一致。
- 但 AIV 的 MTE3 与 Vector/Scalar 重叠执行，属于被隐藏的时间，
  因此 kernel 总时长与端侧耗时都没有变化。

## 精度验证（ATK accuracy，CASE 0~19，混合容差 + CPU FP64 标杆）

| 版本 | 通过 | 失败 |
| --- | ---: | ---: |
| 第一轮基线 | 17/20 | 251 / 259 / 267 |
| 精简写回 | 17/20 | 251 / 259 / 267 |

- 两版逐用例结果**完全一致**（含失败原因：`v_new` 0.9249、`h` 0.9890 通过率）。
- 失败的 3 例均为 `packed_single_export`（`disable_recompute=true`），
  该路径 `storeQG/storeVNew/storeH=true`，按 tiling 规则
  `fusePostWuIntoFwdH=false`，**不会走到本次修改的分支**，属于既有问题。
- 命中本次修改分支的 `packed_single_recompute` 用例（250 / 258 / 266，
  T=1024/1536/2048）全部通过。
- Case 250 的 `attn_out/Aqk/Akk` 在 packed 与 dense 元数据下，
  修改前后**逐位一致**（`torch.equal` 为 True）。

## 结论

1. 改动按要求落地：写回量 48 MiB → 0.75 MiB，MTE3 时间 -16 us，语义等价。
2. **端侧耗时无收益**（784.62 us → 785.93 us，+0.17%，落在噪声范围）。
3. 原因明确：Case 250 的 AIV MTE3 与 Vector/Scalar 重叠，写回被隐藏；
   该 kernel 的瓶颈不在 AIV 写回，而在 AIC 侧（`aic_scalar_time` 约 286 us）
   与同步/流水开销。
4. 后续若继续优化 Case 250，应优先压缩 AIC scalar 与 chunk 间同步等待，
   而不是继续削减 UB→GM 写回流量。
