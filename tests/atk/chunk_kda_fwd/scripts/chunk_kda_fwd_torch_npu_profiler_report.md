# chunk_kda_fwd Case 250 ATK 性能报告

## 口径

- ATK 26.9.8 `performance_device`
- 每个用例采集 20 次
- 指标为 device task `Task Duration(us)` 总和除以 20
- 启动 ATK 前必须 source `fla_npu` 的 `set_env.bash`，确保自定义 OPP 在 CANN op 发现前可见

## 性能对比

| Case | 路径 | 优化前 (us) | 优化后 (us) | 提升 |
| --- | --- | ---: | ---: | ---: |
| 250 | packed `cu_seqlens=[0,1024]` | 1327.80685 | 787.6572 | 40.68% |

## 采样计数核对

| 版本 | API 调用次数 | `ChunkKdaFwd` task 数 | 每次 L0 提交数 | 计数关系 |
| --- | ---: | ---: | ---: | --- |
| 优化前 | 20 | 80 | 4 | 80 = 4 × 20 |
| 优化后 | 20 | 20 | 1 | 20 = 1 × 20 |

优化前、优化后的 `Transpose Count` 均为 20，与 API 调用次数一致。

## 结论

- 优化后 Case 250 为 `787.6572 us`，相对原始基线 `1327.80685 us` 提升 `40.68%`。
- 任务书目标区间为 `1257.5~1191.3 us`，优化后结果低于该区间上限。
- 优化后每次调用只提交 1 个 `ChunkKdaFwd` L0，不再执行 4 段 launch。
