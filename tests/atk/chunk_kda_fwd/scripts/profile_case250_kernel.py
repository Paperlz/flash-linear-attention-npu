#!/usr/bin/env python3
"""Launch the verified Case 250 inputs for external msopprof collection."""

from __future__ import annotations

import argparse
import sys

import torch
import torch_npu

from benchmark_first_round import build_case_250_inputs, run_once


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be positive")

    torch.npu.set_device(args.device)
    q, k, v, g, beta, a_log, dt_bias = build_case_250_inputs(args.device)
    cu_seqlens = [0, 1024]
    for _ in range(args.iterations):
        run_once(q, k, v, g, beta, a_log, dt_bias, cu_seqlens)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
