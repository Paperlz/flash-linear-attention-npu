# ChunkKdaFwd ATK 工程

本目录提供 `chunk_kda_fwd` 的 ATK 单算子工程。通用版本要求、case 范围、测试动作和
结果检查规则见 [`../README.md`](../README.md)。

## 输入限制

- `layout` 支持 `BSND/BNSD/TND/NTD`；`layout` 只描述输入，输出布局由接口固定约定。
- `BSND` 下 `q/k=[B,T,H_k,K]`，`v=[B,T,H_v,V]`，`g=[B,T,H_v,K]`，`beta=[B,T,H_v]`。
- `BNSD` 下 `q/k=[B,H_k,T,K]`，`v=[B,H_v,T,V]`，`g=[B,H_v,T,K]`，`beta=[B,H_v,T]`。
- `TND/NTD` 使用打包 token；`cu_seqlens` 从 `0` 开始、以总 token 数结束且单调不减。
- head 映射满足 `0 < H_k <= H_v <= 128` 且 `H_v % H_k == 0`。
- `q/k/v` 支持 `BFLOAT16/FLOAT16`；`g` 支持 `FLOAT/BFLOAT16`；`beta` 支持 `FLOAT/BFLOAT16`。
- `K/V` 为 `[16,256]` 内 `16` 的倍数，交付矩阵重点覆盖 `K=128`、`V=128/256`。
- `chunk_size` 支持 `64/128`；rank-4 变长输入要求 `B=1`，逻辑序列数最多 `1024`。
- `use_gate_in_kernel=true` 时必须提供 `A_log`；`safe_gate=true` 时 `lower_bound` 取 `[-5,0)`。
- `initial_state` 如提供，末两维由 `state_v_first` 解释为 `[K,V]` 或 `[V,K]`。

## 精度拓扑

精度标准统一使用 `mixed_tolerance_bm`。显式 CPU 节点调用 executor 的 FP64 路径生成
唯一 golden，显式 NPU 节点只运行 DUT。CPU golden 输出转换为 FP32，NPU 输出保持算子
原始 dtype，由 ATK 按混合容差规则比较。

```text
ATK accuracy task
|-- CPU FP64 golden (output FP32)
`-- NPU DUT
```

## 运行

在仓库根目录准备好 ATK、CANN、OPP 和 Python 包环境后执行：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

只验证前 5 个 case：

```bash
CASE_START=0 CASE_END=5 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

性能、确定性、mssanitizer 和用例生成均通过统一脚本的对应 `-scope` 执行。

## K/score 缓冲区复用回归

原始 Case 251、259、267 及更长单序列导出用例覆盖最终 kg 的原始 K 读取。
额外的 12 条用例覆盖 BF16/FP16、四种布局、GVA、tail、chunk=128/V=256，
并保留 initial_state 与 final_state 的检查：

```bash
python tests/atk/chunk_kda_fwd/scripts/gen_kg_lifetime_cases.py \
  <case_dir>/atk_chunk_kda_fwd.json

cd tests/atk/chunk_kda_fwd
atk node --name npu_dut --backend npu --devices 0 --output_path <output_dir> \
  node --name cpu_golden --backend cpu --output_path <output_dir> \
  task -c <case_dir>/atk_chunk_kda_fwd.json --task accuracy --bm_device cpu \
  -p ./executor_chunk_kda_fwd.py --gm_init_flag -mt 1 -to 14400
```

该生成器保持 Case 251 的模型输入分布和混合容差，改变的用例参数均记录在 JSON。
验证结果和已知问题见 [第五轮报告](scripts/chunk_kda_fwd_round5_kg_lifetime_report.md)。

Case 250 的恢复复测、设备任务分解及当前瓶颈见 [第七轮记录](scripts/chunk_kda_fwd_round7_perf_resume_report.md)。

正常执行 PMU、部分核时间与下一轮优化候选见 [第八轮 profiling 报告](scripts/chunk_kda_fwd_round8_profile_report.md)。

尾部V切片候选的三组A/B、撤回结论和边界复现见 [第九轮实验报告](scripts/chunk_kda_fwd_round9_vslice_report.md)。


## R31 提交测试集

按用户要求固定为 20 个原始用例（ID250–269）和 20 个代表性泛化用例，不再执行维度笛卡尔积。
atk_chunk_kda_fwd_fixed20.json 与 atk_chunk_kda_fwd_general20.json 合计 40 例，默认精度 JSON 即二者合并。
泛化集覆盖两种 q dtype、八种 q/g/beta dtype 组合、chunk64/128、BSND/BNSD/TND/NTD、GVA、
短尾块、变长序列、可选状态、raw/preactivated gate；保留已知失败场景，不能把缩减规模记作精度通过。

任务书 §2.2 的中间输出 FP32 与参考实现存在差异，按用户确认保留参考 BF16/FP16 接口；最终状态 FP32。
官方精度复核采用 AscendOpTest 原始 compare_default，并逐个比较全部公开输出。

~~~bash
python tests/atk/chunk_kda_fwd/scripts/gen_legal_domain_cases.py --output-dir /tmp/kda-cases --update-default
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -soc=ascend950 -scope=accuracy
python tests/atk/chunk_kda_fwd/scripts/compare_ascend_op_test.py --atk-output <ATK输出目录> --ascend-op-test <AscendOpTest仓库> --report compare.json
~~~

精度、性能、确定性、内存检测分项结果以交付 reports 和 evidence 为准；未完成或失败项不得记作通过。

## R32 验证（2026-10-17）

被测分支 `feat/chunk-kda-fwd-precision`，以 R31 Fix5 完整包为对照；接口、CPU 标杆、输入种子和值域未改。
Ascend950PR（28 AIC/56 AIV）、CANN 9.1.0-beta.3、Python3.12、torch2.7.1、torch_npu2.7.1.post8。
完整 wheel 构建成功，安装的 373 个 OPP 文件与 wheel 逐字节一致。

ATK 显式 CPU/NPU 节点保存两侧输出，再由未修改的 AscendOpTest `compare_default` 逐输出复核；
golden 按其原始加载规则转为 DUT dtype。阈值 FP32 `[1e-4,1e-4,0.1]`、FP16 `[0.001,0.001,0.1]`、BF16 `[0.004,0.004,0.1]`。
各集合预期 ID 与实际 ID 完全一致，CPU/NPU 输出文件成对，无缺失；以下集合存在重叠，不相加计数。

| 已有测试集 | 用例 | 公开输出 | 失败输出 |
| --- | --- | --- | --- |
| 固定250–269 | 20/20 | 130 | 0 |
| 代表性泛化 | 20/20 | 240 | 0 |
| 历史回归 | 465/465 | 5580 | 0 |
| 原失败定向复测 | 12/12 | 144 | 0 |

修复覆盖：3554 FP16 score；5114/5123 K256 Finalize；5115/5116/5124/5125 窄V打包；
5168/5249/5250/5251 空序列状态编号；5100 小K宽V WH。未执行维度笛卡尔积，也不将465写成496。
精度复核命令沿用上节入口；原始 JSON、比较报告和原失败12例两侧输出已保存于本机 R32 证据包。

同设备串行对照、完整 wheel、100 次原始 Case250 设备采样：

| Case | 形状与 dtype | R31 | R32 | R32/R31 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 250 | B1/T1024/H96/K128/V128，BF16/FP32/BF16，chunk64 | 主核560.72866 μs；加Transpose569.11638 μs | 主核556.02037 μs（std3.87098）；加Transpose564.49423 μs | 0.99188（含Transpose） | 保持目标性能 |

采样口径为 `msprof op_summary` 的 Task Duration；IsFinite/ReduceAll 属测试校验，保留原始 CSV 而不计入算子加Transpose。
任务书历史标杆1323.737 μs、降低10%目标1191.36 μs；当前设备564.49423 μs低于目标，历史标杆未在本机重新构建测量。
性能模型集1/1达到该目标。50轮 `accuracy_dc`：既有 MSS 六例250/2000/2003/2012/2015/5108均通过，
XLSX确定性列为真；工具比较所有输出的 `torch.equal` 与相同NaN位置，不能等同于内存位模式比较。
全block memcheck仍在插桩编译，尚未计通过；阶段05正式验收待此项完成。
