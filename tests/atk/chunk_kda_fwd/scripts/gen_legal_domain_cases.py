"""Reproduce the agreed 20 fixed + 20 representative cases."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser()
p.add_argument("--output-dir",type=Path,required=True)
p.add_argument("--update-default",action="store_true")
a=p.parse_args(); root=Path(__file__).resolve().parents[1]
a.output_dir.mkdir(parents=True,exist_ok=True)
all_cases=[]
for group in ("fixed20","general20"):
 cases=json.loads((root/("atk_chunk_kda_fwd_"+group+".json")).read_text())
 (a.output_dir/(group+".json")).write_text(json.dumps(cases,indent=2)+"\n")
 all_cases.extend(cases)
if a.update_default:
 (root/"atk_chunk_kda_fwd.json").write_text(json.dumps(all_cases,indent=2)+"\n")
print("fixed=20, general=20")
