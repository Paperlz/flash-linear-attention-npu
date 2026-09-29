# chunk_kda_fwd 第四轮优化：W 回传 L1 批量化

## 目标

FwdH 融合路径里，AIV 把 W 从 UB 写进 L1 时用了 8 次 DataCopy（每次按 16 列
gather：`blockCount=32, blockLen=1, srcGap=7`）。同一函数的 vnew 已经采用
"Cast 时整理成 L1 operand 布局 + 一次 DataCopy" 的写法。本轮把 W 也改成这个
写法，缩短 W → L1 → STATE_READY 这段依赖链，同时保留 W 的提前发布与原有
BF16 舍入。

## 代码修改

文件：`op_kernel/arch35/chunk_kda_fwd_fwd_h.h`

- 新增两个私有辅助函数：
  - `CastToL1OperandLayout`：把 32×128 fp32 tile 按 16 列 fractal 整理成
    bf16 L1 operand 布局（与 vnew 原来的写法完全一致）。
  - `CopyOperandTileToL1`：一次 DataCopy（`blockCount=8, blockLen=32,
    srcGap=0, dstGap=32`）把整块 tile 放进 L1。
- W 路径：原先的 `Cast(CAST_RINT)` + 8 次 DataCopy 换成
  `CastToL1OperandLayout` + `CopyOperandTileToL1`，**1 次**搬运。
- vnew 路径：等价重构为同一对辅助函数（参数与原来逐字一致，语义不变）。
- 未改动同步结构：W 仍在 Cast 后立即 `SetDirectFreeAiv()`，仍是
  `V_MTE3` → 单次 DataCopy → `MTE3_V` → `STATE_READY`，W 的提前发布保持不变。

## 精度与稳定性

- Case 250 `attn_out/Aqk/Akk`（packed 与 dense 元数据）与本轮基线
  **逐位一致**（`torch.equal` 全 True）。
- ATK accuracy（CASE 0~19，混合容差 + CPU FP64 标杆）：17/20 通过，
  失败用例与本轮基线**完全相同**（251/259/267，均为 `*_export`，不走该分支）。
- ATK determinism（Case 250，50 次循环）：**Pass**。
- 期间遇到一次 ATK 基础设施故障（compare worker 的 kombu 队列
  `JSONDecodeError` 导致该次 accuracy 卡死），清理后重跑正常，与算子无关。

## 同口径 A/B（ATK `performance_device`，顺序对调交替）

基线 = 云端当前版本（第二轮 gk 精简写回），共 6 组、每组两个版本各跑一次，
基线先/优化先各 3 组以抵消测量顺序偏差。

| 组 | 基线 (us) | 批量 W (us) | 该组先跑 |
| --- | ---: | ---: | --- |
| 1 | 784.17935 | 767.00580 | 基线 |
| 2 | 786.13995 | 769.11845 | 批量 W |
| 3 | 784.99250 | 769.32545 | 基线 |
| 4 | 786.61840 | 767.70730 | 批量 W |
| 5 | 784.98545 | 768.28070 | 批量 W |
| 6 | 784.19415 | 770.08520 | 基线 |
| 均值 | **785.185** | **768.587** | — |
| 标准差 | 0.90 | 1.03 | — |

**提升 −16.60 us（−2.11%）**，两组分布完全不相交（基线 784.2~786.6，
批量 W 767.0~770.1），顺序对调后结论不变。

## 机制分析（Profiler）

MIX_AIC 行 20 次采样均值：

| 指标 (us) | 基线 | 批量 W | 变化 |
| --- | ---: | ---: | ---: |
| kernel `Duration` | 700.5 | 683.6 | **−16.9** |
| `aiv_mte3_time` | 135.5 | 115.8 | **−19.7** |
| `aiv_scalar_time` | 135.5 | 130.2 | −5.3 |
| `aiv_vec_time` | 320.9 | 328.0 | +7.1 |
| `aiv_mte2_time` | 105.9 | 105.6 | −0.3 |
| `aic_scalar_time` | 285.0 | 284.9 | −0.1 |
| `aic_mac_time` | 105.7 | 105.6 | −0.1 |

- 收益来源：W 的 L1 搬运从 8 条 strided 小 DataCopy（每条只搬 1 KiB、
  源侧跳 7 个 block）变成 1 条连续 8 KiB 的 DataCopy，MTE3 忙时直降 19.7 us，
  同时标量下发次数减少 5.3 us。
- 代价：strided Cast 比原来的整块连续 Cast 多花 7.1 us 向量时间，
  净收益仍有 16.9 us。
- AIC 侧完全不变，说明收益全部来自 AIV 侧 W 回传段的缩短。

## 结论与决策

1. W 的 L1 搬运确实在依赖链上：批量化后单次 −2.11%，是目前除第一轮外
   唯一一次可测的正收益。
2. 决策：**保留本轮修改**，当前最优版本更新为「第二轮 gk 精简写回 +
   第四轮 W 回传批量化」。
3. 剩余空间：W 的 strided Cast 仍有 7.1 us 向量开销，若能进一步压缩
   （例如与 U 的 Cast 合并成一次遍历），可继续扩大收益。
