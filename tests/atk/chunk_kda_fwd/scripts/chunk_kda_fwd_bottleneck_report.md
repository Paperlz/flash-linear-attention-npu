# ChunkKdaFwd Case 250 融合路径瓶颈测试

## 测试范围

- Case：ATK 250
- Shape：`B=1, T=1024, H=96, K=128, V=128`
- 输入：`BSND`，`q/k/v=BF16`，`g=FP32`，`beta=BF16`
- 属性：`chunk_size=64, safe_gate=true, use_gate_in_kernel=true`
- 路径：第一轮优化后的 A5 dense 单 launch 融合路径
- 整核采集：`msopprof`
  - `BasicInfo`
  - `PipeUtilization`
  - `Memory`
  - `Occupancy`
- 分段探针：临时在 `RunFrontEnd` 与 `fwdH.Process` 两侧读取 `SYS_CNT`，只用于定位，已回退

## 1. ChunkKdaFwd 与 Transpose

ATK 20 次采样结果：

| device task | Count | Total Time(us) | per launch(us) | 占比 |
| --- | ---: | ---: | ---: | ---: |
| ChunkKdaFwd | 20 | 14065.552 | 703.278 | 98.9% |
| Transpose | 20 | 159.414 | 7.971 | 1.1% |

结论：剩余设备侧耗时几乎全部落在 `ChunkKdaFwd`，`Transpose` 不是本轮优化对象。

ATK `performance_device` 为 `787.6572 us`，与两个 device task 合计约
`711.25 us` 的差值主要来自 launch、dispatch 和其他非目标 device task。

## 2. Prepare 与融合 FwdH/输出

`msopprof BasicInfo`：

```text
Op Type: mix
Task Duration: 704.540 us
Block Dim: 28
Mix Block Dim: 56
Current Freq: 1650 MHz
```

临时 `SYS_CNT` 分段结果按约 1 GHz 换算，与整核 `704.5 us` 对齐：

| 阶段 | AIC 样本范围 | AIV 样本范围 | 结论 |
| --- | ---: | ---: | --- |
| Prepare (`RunFrontEnd`) | 441.6–443.1 us | 441.7–443.1 us | 各核均匀，约占整核 63% |
| 融合 FwdH/输出 (`fwdH.Process`) | 191.9–252.4 us | 189.7–249.4 us | 明显分档 |
| 两段合计 | 633.9–694.0 us | 631.6–691.2 us | 与 704.5 us 整核吻合 |

融合 FwdH/输出呈两档：

- 0–11 号物理核：约 `245–252 us`
- 12–27 号物理核：约 `190–203 us`

两档差约 `50–55 us`，与 `msopprof Occupancy` 对 0–11 号核偏慢的判断一致。

## 3. Pipe 利用率

代表性单核均值：

| 核类型 | Pipe | 活跃时间(us) | 占比 |
| --- | --- | ---: | ---: |
| AIC | Cube | 62.1 | 8.9% |
| AIC | Scalar | 282.3 | 40.8% |
| AIC | MTE1 | 51.2 | 7.4% |
| AIC | MTE2 | 204.9 | 29.6% |
| AIC | Fixpipe | 218.2 | 31.5% |
| AIV | Vector | 324.8 | 46.9% |
| AIV | Scalar | 134.0 | 19.4% |
| AIV | MTE2 | 115.7 | 16.7% |
| AIV | MTE3 | 154.1 | 22.3% |

结论：

- 没有单一 Pipe 超过 80%，不是纯 Cube 或纯 Vector 饱和。
- AIC 以 Scalar、MTE2、Fixpipe 为主，AIV 以 Vector、MTE3 为主。
- 性能报告提示 compute usage < 20%，MTE1/MTE2 带宽利用率低于 80%；
  主要问题是流水等待、同步和核间负载不均，而不是单点算力打满。

## 4. 对 `gk` 精简写回候选的判断

代码事实：

- Prepare 将完整 FP32 `gk` 写回 GM，Case 250 逻辑写量为 `48 MiB`。
- FwdH 每个 chunk 只读取最后一行，共约 `0.75 MiB`。
- 理论上可减少约 `47.25 MiB` 写回。

本轮证据：

- `gk` 写回位于 Prepare；Prepare 在各核均为约 `442 us`，不存在明显核间失衡。
- AIV MTE3 占比约 22%，`gk` 是其中一部分而非唯一写流量。
- 当前 FwdH 尾核比快核多约 `50–55 us`，已经是可量化的负载不均来源。

判断：

- `gk` 精简写回仍然是低风险、易隔离的 A/B 候选，值得单独验证。
- 但从当前占比估计，它更可能贡献十几到几十微秒，单独达到“再降 5%
  = 约 39.4 us”存在不确定性。
- 若本轮只选一个改动，优先处理 FwdH/输出在 0–11 号核上的
  `50–55 us` 尾巴收益上限更高；`gk` 精简写回可作为并行或紧随其后的
  第二轮候选。

## 采集产物

- 整核指标：
  `/tmp/kda_case250_msopprof_pipe.I52toQ/OPPROF_20260909233623_WKUONNCGGVZYBMMI`
- BasicInfo：
  `/tmp/kda_case250_msopprof_basic.U60Ioz/OPPROF_20260909233411_RGHKKKYNTCUEAGJL`

## 复测口径

正式性能结论仍使用 ATK `performance_device`：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
FLA_NPU_ENV=/usr/local/python3.12.13/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash \
CASE_START=0 CASE_END=1 \
ATK_OUTPUT_ROOT=/tmp/kda_atk_perf_case250_env \
bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd -npu_device_id=0 -soc=ascend950 -scope=performance
```
