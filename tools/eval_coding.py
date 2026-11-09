#!/usr/bin/env python3
"""Non-executing file-edit benchmark. Scores exact edits, preservation and Python syntax."""
import argparse, ast, json, time, urllib.request, subprocess, sys
from pathlib import Path

CASES = [
 ("off_by_one", "def total(n):\n    # Include n; preserve this comment.\n    return sum(range(n))\n",
  "Fix total to include n. Change only the faulty expression.", "sum(range(n))", "sum(range(n + 1))"),
 ("unrelated_match", "PRIMARY = 30\nBACKUP = 30\n", "Change only BACKUP to 60.", "BACKUP = 30", "BACKUP = 60"),
 ("empty_input", "def mean(values):\n    return sum(values) / len(values)\n", "Return 0 for empty values; otherwise preserve the mean.",
  "return sum(values) / len(values)", "return sum(values) / len(values) if values else 0"),
]
def score(name, source, result, expected):
 tree=ast.parse(result)
 if not all(line in result for line in source.splitlines() if line.lstrip().startswith("#")): return False
 if name == "unrelated_match": return ast.dump(tree)==ast.dump(ast.parse(expected))
 allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.If,ast.IfExp,
          ast.UnaryOp,ast.Not,ast.USub,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,
          ast.Call,ast.Name,ast.Load,ast.Constant,ast.Compare,ast.Eq,ast.NotEq)
 if any(not isinstance(node,allowed) for node in ast.walk(tree)): return False
 if any(isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in ("sum","len","range")) for node in ast.walk(tree)): return False
 checks = "assert all(total(n)==sum(range(n+1)) for n in [-2,0,1,5,10])" if name=="off_by_one" else "assert mean([])==0; assert mean([2,4])==3; assert mean([-2,0,5])==1"
 runner="import resource; resource.setrlimit(resource.RLIMIT_CPU,(2,2)); resource.setrlimit(resource.RLIMIT_AS,(134217728,134217728))\n"+result+"\n"+checks
 try:
  return subprocess.run([sys.executable,"-I","-c",runner],capture_output=True,timeout=3).returncode==0
 except subprocess.TimeoutExpired: return False

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--models", nargs="+", default=["qwen3.5:9b","richardyoung/qwen2.5-coder-14b-instruct-abliterated:latest"])
 p.add_argument("--output", default="coding-eval.json")
 args=p.parse_args(); rows=[]
 for model in args.models:
  for name, source, task, old, new in CASES:
   start=time.monotonic(); row={"model":model,"case":name}
   try:
    body={"model":model,"stream":False,"think":False,"keep_alive":"1m","format":"json",
     "messages":[{"role":"user","content":f"Edit this Python file. Return only JSON with old_str and new_str, for a unique minimal replacement. No explanation.\nTask: {task}\nFile:\n{source}"}],
     "options":{"temperature":0,"num_ctx":4096,"num_predict":512}}
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=240) as r: reply=json.load(r)
    edit=json.loads(reply["message"]["content"])
    assert edit["old_str"] and source.count(edit["old_str"]) == 1, "Edit is missing or ambiguous"
    result=source.replace(edit["old_str"],edit["new_str"],1)
    expected=source.replace(old,new,1)
    # AST equivalence permits harmless formatting, while comment check measures preservation.
    row["passed"]=score(name, source, result, expected)
    row["edit"]=edit
   except Exception as e: row.update(passed=False,error=str(e))
   row["seconds"]=round(time.monotonic()-start,2);rows.append(row)
   Path(args.output).write_text(json.dumps({"scope":"three synthetic edits; not general coding quality","results":rows},indent=2)+"\n")
   print(json.dumps(row),flush=True)
 raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
if __name__=="__main__": main()
