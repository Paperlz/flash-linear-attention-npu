from pathlib import Path
import argparse,sys,json,torch,numpy as np
from ml_dtypes import bfloat16
parser=argparse.ArgumentParser(description="Compare ATK saved outputs with unmodified AscendOpTest defaults.")
parser.add_argument("--atk-output",type=Path,required=True)
parser.add_argument("--ascend-op-test",type=Path,required=True)
parser.add_argument("--report",type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.ascend_op_test / "compare"))
from compare.compare import compare_default
from compare.accuracy_config import AccuracyConfig
torch.set_num_threads(1)
root=args.atk_output;destination=args.report;rows=[]
for npu in sorted(root.rglob("output/npu_npu_dut/*/*/output_*.pt")):
 cpu=Path(str(npu).replace("/npu_npu_dut/","/cpu_cpu_golden/"))
 if not cpu.exists():raise RuntimeError("missing CPU golden: "+str(cpu))
 x=torch.load(npu,map_location="cpu",weights_only=False)
 y=torch.load(cpu,map_location="cpu",weights_only=False)
 dt=str(x.dtype).split(".")[-1]
 if x.shape!=y.shape:raise RuntimeError((str(npu),x.shape,y.shape))
 # AscendOpTest reads golden and output with the declared output dtype.
 # Keep the official file-loading dtype and the unmodified comparison function.
 y=y.to(x.dtype)
 xf=x.float().numpy().reshape(-1);yf=y.float().numpy().reshape(-1)
 finite=bool(np.isfinite(xf).all() and np.isfinite(yf).all())
 threshold=AccuracyConfig.default_acc[dt]
 xn=x.contiguous().view(torch.uint16).numpy().view(bfloat16) if x.dtype==torch.bfloat16 else x.contiguous().numpy()
 yn=y.contiguous().view(torch.uint16).numpy().view(bfloat16) if y.dtype==torch.bfloat16 else y.contiguous().numpy()
 passed=finite and bool(compare_default(xn.reshape(-1).copy(),yn.reshape(-1).copy(),threshold,show_err_num=0))
 diff=np.abs(xf-yf);bad=np.where(np.abs(yf)>=1,diff/(np.maximum(np.abs(xf),np.abs(yf))+1e-9)>threshold[0],diff>threshold[0])
 rows.append({"case":int(npu.parent.name),"output":npu.name,"dtype":dt,"elements":x.numel(),"threshold":threshold,"finite":finite,"pass":passed,"max_abs":float(diff.max()) if diff.size else 0,"bad":int(bad.sum())})
 if not passed:print("FAIL",rows[-1],flush=True)
result={"tool":"AscendOpTest compare_default","source":str(args.ascend_op_test / "compare/compare/compare.py"),"golden_conversion":"output dtype as in official compare() file loading","outputs":len(rows),"cases":len(set(x["case"] for x in rows)),"failed_outputs":sum(not x["pass"] for x in rows),"results":rows}
destination.write_text(json.dumps(result,indent=2))
print({k:v for k,v in result.items() if k!="results"},flush=True)
sys.exit(0 if rows and result["failed_outputs"]==0 else 1)
