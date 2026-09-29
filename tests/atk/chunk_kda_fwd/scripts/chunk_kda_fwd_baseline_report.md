# chunk_kda_fwd Case 250 性能基线

## 正式 ATK 口径

- 设备：Ascend950PR
- 软件：CANN 9.1.0-beta.3、torch 2.7.1、torch_npu 2.7.1.post8、ATK 26.9.8
- 算子版本：`e50df9e` 原始基线
- 用例：任务书 Case 250
  - `B=1, T=1024, H=96, K=128, V=128`
  - `BSND`、`chunk_size=64`
  - `q/k/v=BF16, g=FP32, beta=BF16`
  - `safe_gate=true, use_gate_in_kernel=true`
- 命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
FLA_NPU_ENV=/usr/local/python3.12.13/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash \
CASE_START=0 CASE_END=1 \
bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd -npu_device_id=0 -soc=ascend950 -scope=performance
```

## 采样计数

- ATK 实际执行 20 次性能采集。
- `task_duration` 使用全部 device task 的 `Task Duration(us)` 求和后除以 20。
- 原始 packed 路径每次调用提交 4 个 `ChunkKdaFwd` L0。
- `op_statistic.csv` 中 `ChunkKdaFwd Count=80 = 4 × 20`，`Transpose Count=20`，计数自洽。

## 基线结果

| Case | Shape | DType | device_perf (us) | 任务书标杆 (us) | 偏差 |
| --- | --- | --- | ---: | ---: | ---: |
| 250 | `B=1,T=1024,H=96,K=128,V=128` | BF16/FP32/BF16 | 1327.80685 | 1323.737 | +0.31% |

## 结论

- 本地 ATK 基线复测与任务书标杆相差约 `0.31%`，可以作为后续同口径对比基线。
