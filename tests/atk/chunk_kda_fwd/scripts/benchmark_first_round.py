#!/usr/bin/env python3
"""Profile Case 250 packed-varlen versus dense metadata paths.

The script uses the fixed torch_npu.profiler schedule required by the
AscendC performance-evaluation workflow: warmup=5, active=5, repeat=1.
It reports the sum of all rows in op_statistic.csv divided by active steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

import torch
import torch_npu

from fla_npu.ops.ascendc import chunk_kda_fwd


CASE = {
    "B": 1,
    "T": 1024,
    "H": 96,
    "K": 128,
    "V": 128,
    "chunk_size": 64,
    "seed": 20260812,
}
WARMUP = 5
ACTIVE = 5
REPEAT = 1


def _normal_quantized(
    shape,
    dtype,
    device,
    generator,
    *,
    mean=0.0,
    std=1.0,
    sigmoid=False,
    l2_normalize=False,
):
    value = torch.randn(shape, generator=generator, dtype=torch.float32).mul(std).add(mean)
    if sigmoid:
        value = torch.sigmoid(value)
    if l2_normalize:
        value = value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + 1e-6)
    return value.to(dtype).to(device)


def build_case_250_inputs(device):
    batch = CASE["B"]
    tokens = CASE["T"]
    heads = CASE["H"]
    key_dim = CASE["K"]
    value_dim = CASE["V"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(CASE["seed"])

    q = _normal_quantized(
        (batch, tokens, heads, key_dim),
        torch.bfloat16,
        device,
        generator,
        std=0.05,
        l2_normalize=True,
    )
    k = _normal_quantized(
        (batch, tokens, heads, key_dim),
        torch.bfloat16,
        device,
        generator,
        std=0.05,
        l2_normalize=True,
    )
    v = _normal_quantized(
        (batch, tokens, heads, value_dim),
        torch.bfloat16,
        device,
        generator,
        std=0.05,
    )
    g = _normal_quantized(
        (batch, tokens, heads, key_dim),
        torch.float32,
        device,
        generator,
        std=1.25,
    )
    beta = _normal_quantized(
        (batch, tokens, heads),
        torch.bfloat16,
        device,
        generator,
        mean=1.5,
        std=0.35,
        sigmoid=True,
    )
    a_log = _normal_quantized(
        (heads,),
        torch.float32,
        device,
        generator,
        std=0.12,
    )
    dt_bias = _normal_quantized(
        (heads * key_dim,),
        torch.float32,
        device,
        generator,
        mean=-3.0,
        std=1.65,
    )
    return q, k, v, g, beta, a_log, dt_bias


def run_once(q, k, v, g, beta, a_log, dt_bias, cu_seqlens):
    return chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        CASE["K"] ** -0.5,
        CASE["chunk_size"],
        layout="BSND",
        safe_gate=True,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
    )


def profile_case(name, q, k, v, g, beta, a_log, dt_bias, cu_seqlens, trace_root):
    case_root = trace_root / name
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True, exist_ok=True)
    schedule = torch_npu.profiler.schedule(
        wait=0,
        warmup=WARMUP,
        active=ACTIVE,
        repeat=REPEAT,
    )
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(case_root)),
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(REPEAT * (WARMUP + ACTIVE)):
            run_once(q, k, v, g, beta, a_log, dt_bias, cu_seqlens)
            prof.step()
    torch.npu.synchronize()

    profile_dirs = sorted(
        case_root.rglob("*_ascend_pt/ASCEND_PROFILER_OUTPUT"),
        key=lambda path: path.stat().st_mtime,
    )
    if not profile_dirs:
        raise RuntimeError(f"no ASCEND_PROFILER_OUTPUT produced under {case_root}")
    output_dir = profile_dirs[-1]

    step_path = output_dir / "step_trace_time.csv"
    if not step_path.is_file():
        raise RuntimeError(f"no step_trace_time.csv produced under {output_dir}")
    step_computing_us = []
    with step_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step_computing_us.append(float(row["Computing"]))
    if len(step_computing_us) != ACTIVE * REPEAT:
        raise RuntimeError(
            f"expected {ACTIVE * REPEAT} active steps, got {len(step_computing_us)}"
        )

    op_path = output_dir / "op_statistic.csv"
    if not op_path.is_file():
        raise RuntimeError(f"no op_statistic.csv produced under {output_dir}")
    op_total_us = 0.0
    rows = []
    with op_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("Total Time(us)")
            if value is None:
                continue
            value_f = float(value)
            op_total_us += value_f
            rows.append(
                {
                    "op_type": row.get("OP Type") or row.get("Op Type") or "",
                    "count": row.get("Count") or "",
                    "total_time_us": value_f,
                }
            )

    api_path = output_dir / "api_statistic.csv"
    api_count = None
    if api_path.is_file():
        with api_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("API Name") == "aclnnChunkKdaFwd":
                    api_count = int(row["Count"])
                    break
    return {
        "case": name,
        "per_step_us": statistics.mean(step_computing_us),
        "step_computing_us": step_computing_us,
        "active_step_count": len(step_computing_us),
        "api_count": api_count,
        "op_statistic_total_us": op_total_us,
        "op_statistic_count_includes_warmup_spill": True,
        "csv": str(op_path),
        "rows": sorted(rows, key=lambda row: row["total_time_us"], reverse=True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--trace-root",
        default="/tmp/chunk_kda_fwd_first_round_profile",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    q, k, v, g, beta, a_log, dt_bias = build_case_250_inputs(args.device)
    trace_root = Path(args.trace_root)
    trace_root.mkdir(parents=True, exist_ok=True)

    packed = []
    dense = []
    for repeat in range(args.repeats):
        packed.append(
            profile_case(
                f"packed_r{repeat}",
                q,
                k,
                v,
                g,
                beta,
                a_log,
                dt_bias,
                [0, CASE["T"]],
                trace_root,
            )
        )
        dense.append(
            profile_case(
                f"dense_r{repeat}",
                q,
                k,
                v,
                g,
                beta,
                a_log,
                dt_bias,
                None,
                trace_root,
            )
        )

    def summarize(name, results):
        values = [result["per_step_us"] for result in results]
        return {
            "case": name,
            "runs_us": values,
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "latest_rows": results[-1]["rows"],
        }

    report = {
        "warmup": WARMUP,
        "active": ACTIVE,
        "repeat": REPEAT,
        "case_250": CASE,
        "packed": summarize("packed", packed),
        "dense": summarize("dense", dense),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
