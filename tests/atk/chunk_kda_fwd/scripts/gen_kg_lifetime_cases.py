#!/usr/bin/env python3
"""Generate ATK regressions for K/score lifetime."""

import argparse
import json
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
from gen_chunk_kda_fwd import _case_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output ATK JSON path")
    args = parser.parse_args()
    cases = json.loads((OP_DIR / "atk_chunk_kda_fwd.json").read_text())
    case = next(case for case in cases if case["id"] == 251)
    base = json.loads(next(item["range_values"] for item in case["inputs"]
                           if item["name"] == "case_spec"))
    del base["shape_spec"], base["optional_spec"]
    specs = []
    for dtype in ("bf16", "fp16"):
        for layout in ("BSND", "BNSD", "TND", "NTD"):
            specs.append(dict(base, B=1, H=2, HV=4, T=128, layout=layout,
                              q_dtype=dtype, cu_seqlens="0,128",
                              explicit_chunk_indices=True, initial_state=True,
                              output_final_state=True,
                              state_v_first=layout in ("BSND", "TND")))
        specs.append(dict(base, B=1, H=2, HV=4, T=145, layout="TND", q_dtype=dtype,
                          cu_seqlens="0,17,82,145", explicit_chunk_indices=True,
                          initial_state=True, output_final_state=True, state_v_first=False))
        specs.append(dict(base, B=2, H=2, HV=4, T=256, K=128, V=256, chunk_size=128,
                          layout="BNSD", q_dtype=dtype, cu_seqlens="",
                          explicit_chunk_indices=False, initial_state=True,
                          output_final_state=True, state_v_first=True))
    output = []
    for index, spec in enumerate(specs):
        spec["profile"] = "kg_lifetime_generalization"
        spec["case_key"] = (f"kg_lifetime_{index}_{spec['q_dtype']}_{spec['layout']}_"
                            f"t{spec['T']}_v{spec['V']}")
        output.append(_case_payload(600 + index, spec))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(output)} cases to {args.output}")


if __name__ == "__main__":
    main()
