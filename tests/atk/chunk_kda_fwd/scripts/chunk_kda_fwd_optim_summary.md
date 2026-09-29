# chunk_kda_fwd 第一轮优化总结

## 优化目标

将 `B=1, cu_seqlens=[0,T], T % chunk_size == 0` 的单序列 packed 调用在 L2 内部
归一化为 dense 元数据，使 Case 250 复用 A5 单 launch 融合路径，减少中间量 GM
往返和 L0 提交次数。

## 代码修改

修改文件：

`fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp`

修改内容：

- 在原始参数校验完成后识别单序列 full-chunk 快路径。
- 仅对 BF16、`chunk_size=64`、`K=V=128`、`seqlen % chunk_size == 0` 的调用生效。
- 向内部 L0 传入 `nullptr` 形式的 `cu_seqlens/chunk_indices`，使 launch 选择和 host
  tiling 识别为 dense。
- 公开输入、输出 shape、dtype、layout 和返回值契约保持不变。

## 正确性验证

- ATK 原始 Case 250 精度：`1/1 Pass`。
- Case 250 优化后 packed 与 dense 的 `attn_out/Aqk/Akk` 逐位一致。
- BSND/BNSD/TND/NTD 的单序列 full-chunk 小用例均与 dense 路径逐位一致。
- A5 多 chunk 回归：
  `test_chunk_kda_fwd_a5_multichunk_all_outputs`，`2 passed in 5.71s`。
- 单序列 full-chunk CPU 参考检查：
  `attn_out` 最大绝对误差 `7.92e-05`，cosine `0.99996`；
  `Aqk/Akk` 最大绝对误差 `6.11e-05`。

## 性能结果

| Case | 优化前 (us) | 优化后 (us) | 提升 |
| --- | ---: | ---: | ---: |
| 250 packed | 1327.80685 | 787.6572 | 40.68% |

正式结果来自 ATK `performance_device`，不是本地固定 5 次 active-step profiler。

## 采样计数核对

- ATK 采集 20 次。
- 优化前 `ChunkKdaFwd Count=80`，对应 4 阶段 × 20 次。
- 优化后 `ChunkKdaFwd Count=20`，对应单 launch × 20 次。
- 之前 `Total Time(us)/5=52.49%` 的数值无效，因为 `op_statistic.Count=7` 包含了 2 个
  warmup 尾部 task，不能除以 5。

## 结论

- 第一轮优化方向有效，ATK 同口径提升 `40.68%`，达到并超过任务书 `5%~10%` 目标。
- packed 优化后达到 dense 单 launch 路径性能，主要收益来自减少 3 次 L0 提交和中间量往返。
- 当前归一化条件覆盖所有 layout，但只对单序列 full-chunk 且 A5 快路径 shape 生效，
  不改变普通变长和多序列路径。
