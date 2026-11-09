#!/usr/bin/env python3
"""Run deterministic coding, memory, and recovery regressions; optional model comparisons."""
import argparse,json,subprocess,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--models",action="store_true");a=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 checks=[
  ("coding_and_memory",["cargo","test","--quiet","--","--test-threads=1"],root),
  ("repair_faults",["python3","-m","unittest","discover","-s",str(root.parent/"recovery-kit"),"-v"],root),
 ]
 if a.models:
  checks += [(name,["python3",str(root/"tools"/script)],root) for name,script in [("coding_models","eval_coding.py"),("recovery_models","eval_recovery.py")]]
 results=[]
 for name,argv,cwd in checks:
  start=time.monotonic()
  try:
   run=subprocess.run(argv,cwd=cwd,capture_output=True,text=True,timeout=900)
   row={"name":name,"passed":run.returncode==0,"exit_code":run.returncode,"output":(run.stdout+run.stderr)[-20000:]}
  except subprocess.TimeoutExpired:row={"name":name,"passed":False,"error":"900-second deadline exceeded"}
  row["seconds"]=round(time.monotonic()-start,2);results.append(row)
  (root/"workflow-eval.json").write_text(json.dumps({"results":results},indent=2)+"\n")
  print(json.dumps({"name":name,"passed":row["passed"],"seconds":row["seconds"]}),flush=True)
 raise SystemExit(0 if all(r["passed"] for r in results) else 1)
if __name__=="__main__":main()
