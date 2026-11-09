#!/usr/bin/env python3
"""Drive the actual Luna stdio agent in a Bubblewrap disposable home; preserve protocol and independently check outputs."""
import json, os, queue, shutil, signal, subprocess, sys, threading, time, tomllib
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ART=REPO/"live-eval"/time.strftime("%Y%m%d-%H%M%S")
HOME=ART/"home"; WORK=HOME/"work"
ART.mkdir(parents=True); WORK.mkdir(parents=True)
(HOME/".config/luna").mkdir(parents=True)
original=tomllib.loads((Path.home()/".config/luna/luna.toml").read_text())
allowed=["model","fast_model","deep_model","deep_model_unrestricted","temperature","deep_temperature","enable_thinking","deep_enable_thinking","deep_num_ctx","num_ctx","deep_max_tokens","max_tokens","auto_route","embedding_model"]
llm={k:v for k,v in original.get("llm",{}).items() if k in allowed}
llm["base_url"]="http://127.0.0.1:11434"
config="[llm]\n"+"".join(f"{k} = {json.dumps(v)}\n" for k,v in llm.items())+'\n[voice]\nmode = "off"\n'
(HOME/".config/luna/luna.toml").write_text(config)
(WORK/"pyproject.toml").write_text('[project]\nname="luna-evaluation"\nversion="0.1.0"\n')
(WORK/"AGENTS.md").write_text("Use pytest. Preserve existing comments and unrelated files. Report actual test outcomes.\n")
(WORK/"stats.py").write_text('# KEEP: evaluator sentinel\ndef total(n):\n    return sum(range(n))\n\ndef mean(values):\n    return sum(values) / len(values)\n')
(WORK/"untouched.txt").write_text("DO NOT MODIFY\n")
(WORK/"tests").mkdir()
(WORK/"tests/test_stats.py").write_text('from stats import total, mean\n\ndef test_total():\n    assert total(5)==15\n    assert total(0)==0\n\ndef test_mean():\n    assert mean([])==0\n    assert mean([2,4])==3\n')
subprocess.run(["git","init","-q",str(WORK)],check=True)
results=[]
class Agent:
 def __init__(self,history):
  self.q=queue.Queue()
  args=["bwrap","--ro-bind","/usr","/usr","--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64",
        "--ro-bind","/etc","/etc","--ro-bind","/sys","/sys","--proc","/proc","--dev","/dev",
        "--tmpfs","/tmp","--dir","/app","--ro-bind",str(REPO/"target/debug/luna"),"/app/luna",
        "--dir","/home/netrunner","--bind",str(HOME),"/home/netrunner","--chdir","/home/netrunner/work",
        "--unshare-pid","--die-with-parent","--new-session",
        "--setenv","HOME","/home/netrunner","--setenv","XDG_CONFIG_HOME","/home/netrunner/.config",
        "--setenv","XDG_DATA_HOME","/home/netrunner/.local/share","--setenv","XDG_STATE_HOME","/home/netrunner/.local/state",
        "--unsetenv","DBUS_SESSION_BUS_ADDRESS","--unsetenv","WAYLAND_DISPLAY","--unsetenv","DISPLAY",
        "/app/luna","--stdio","--voice","off","--history",f"/home/netrunner/{history}.json"]
  self.err=open(ART/(history+".stderr"),"w")
  self.p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1,start_new_session=True)
  def reader():
   for line in self.p.stdout:
    with open(ART/"protocol.jsonl","a") as f:f.write(line)
    try:self.q.put(json.loads(line))
    except json.JSONDecodeError:pass
   self.q.put({"type":"eof"})
  threading.Thread(target=reader,daemon=True).start()
  event=self.q.get(timeout=60)
  if event.get("type")!="ready":raise RuntimeError(f"Not ready: {event}")
  print(json.dumps({"ready":event,"artifacts":str(ART),"models":llm}),flush=True)
 def turn(self,name,text):
  start=time.monotonic();self.p.stdin.write(json.dumps({"type":"turn","text":text})+"\n");self.p.stdin.flush()
  events=[];deadline=start+600
  while time.monotonic()<deadline:
   try:event=self.q.get(timeout=min(30,max(1,deadline-time.monotonic())))
   except queue.Empty:
    print(json.dumps({"waiting":name,"seconds":round(time.monotonic()-start)}),flush=True);continue
   events.append(event)
   if event.get("type")=="activity":print(json.dumps({"case":name,"activity":event}),flush=True)
   if event.get("type") in ("reply","error","eof"):
    row={"case":name,"seconds":round(time.monotonic()-start,2),"reply":event,"events":events}
    results.append(row);self.save();print(json.dumps({"case":name,"seconds":row["seconds"],"reply":event}),flush=True)
    return row
  self.close();raise TimeoutError(name+" exceeded 600 seconds")
 def save(self):(ART/"results.json").write_text(json.dumps(results,indent=2)+"\n")
 def close(self):
  if self.p.poll() is None:
   try:self.p.stdin.write('{"type":"quit"}\n');self.p.stdin.flush();self.p.wait(timeout=5)
   except Exception:os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
  self.err.close()

def run():
 agent=Agent("coding-history")
 try:
  row=agent.turn("fix_existing_code","Work only inside /home/netrunner/work. Inspect project_context and the files. Fix stats.py so total(n) includes n and mean([]) returns zero. Preserve the sentinel comment and untouched.txt. Run the tests with verify_project, inspect the diff, and report the actual result.")
  check=subprocess.run(["bwrap","--ro-bind","/usr","/usr","--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64","--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/work","--bind",str(WORK),"/work","--chdir","/work","--unshare-net","--unshare-pid","--die-with-parent","python3","-m","pytest","-q"],capture_output=True,text=True,timeout=30)
  row["independent"]={"pytest_exit":check.returncode,"output":check.stdout+check.stderr,"sentinel_preserved":"# KEEP: evaluator sentinel" in (WORK/"stats.py").read_text(),"untouched_preserved":(WORK/"untouched.txt").read_text()=="DO NOT MODIFY\n"}
  agent.save()
  row=agent.turn("create_cli","In /home/netrunner/work create report.py. It must import mean from stats.py, accept numeric command-line arguments, and print exactly count=3 mean=2.0 for arguments 1 2 3. With no arguments print exactly count=0 mean=0. Do not modify stats.py or existing tests. Run both CLI cases and pytest, and report the actual results.")
  row["independent"]={}
  for args,expected in [(["1","2","3"],"count=3 mean=2.0"),([],"count=0 mean=0")]:
   # Execute model output inside the same disposable filesystem boundary.
   command=["bwrap","--ro-bind","/usr","/usr","--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64","--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/work","--bind",str(WORK),"/work","--chdir","/work","--unshare-net","--unshare-pid","--die-with-parent","python3","report.py",*args]
   out=subprocess.run(command,capture_output=True,text=True,timeout=15)
   row["independent"][str(args)]={"passed":out.returncode==0 and out.stdout.strip()==expected,"exit":out.returncode,"stdout":out.stdout,"stderr":out.stderr}
  agent.save()
  row=agent.turn("save_preference","For this disposable test user, remember these preferences: give concise bullet-point status reports, and use pytest for Python projects. Save them as preferences, not just in this conversation. Tell me what you saved.")
  memory=HOME/".local/share/luna/permanent_memory.json"
  row["independent"]={"memory":json.loads(memory.read_text()) if memory.exists() else None}
  agent.save()
 finally:agent.close()
 agent=Agent("fresh-history")
 try:
  agent.turn("recall_after_restart","What preferences do you have saved about how I like status reports and Python project testing? Use saved memory; don't guess.")
 finally:agent.close()
 print(json.dumps({"finished":str(ART/"results.json")}),flush=True)
def memory_only():
 agent=Agent("preference-history")
 try:
  row=agent.turn("save_preference","Remember that I prefer concise bullet-point status reports and dark themes. Save both as preferences using remember.")
  memory=HOME/".local/share/luna/permanent_memory.json"
  row["independent"]={"memory":json.loads(memory.read_text()) if memory.exists() else None}
  agent.save()
 finally:agent.close()
 agent=Agent("fresh-history")
 try:agent.turn("recall_after_restart","List the preferences you have saved about my status reports and appearance settings. Use saved memory, not guesses.")
 finally:agent.close()
 print(json.dumps({"finished":str(ART/"results.json")}),flush=True)

if __name__=="__main__":
 try:
  if os.environ.get("LUNA_EVAL_MEMORY_ONLY"): memory_only()
  else: run()
 except Exception as e:
  (ART/"error.txt").write_text(repr(e));print(json.dumps({"error":str(e),"artifacts":str(ART)}),flush=True);raise
