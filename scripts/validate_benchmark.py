import os, sys, json, time, argparse
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.benchmark_datasets import load_funsd, load_iiit5k, cer, norm
from chorus import pipeline

def run(dataset,n):
    data=load_funsd(n) if dataset=="funsd" else load_iiit5k(n)
    rows=[]; exact=0; cer_sum=0.0; sec_sum=0.0
    print(f"{dataset}: {len(data)} Chorus dogrulama ornegi",flush=True)
    for idx,(img,ref) in enumerate(data,1):
        t0=time.time(); err=""
        try:
            out=pipeline.read(img)
            hyp=norm(out.get("text","")); route=out.get("route","")
        except Exception as ex:
            hyp=""; route="error"; err=repr(ex)
        dt=time.time()-t0; rl=ref.lower(); hl=hyp.lower(); c=cer(rl,hl)
        exact += int(rl==hl); cer_sum += c; sec_sum += dt
        rows.append({"i":idx,"ref":ref,"hyp":hyp,"exact":rl==hl,"cer":round(c,4),"sec":round(dt,2),"route":route,"error":err})
        print(f"[{idx}/{len(data)}] {route} ref={ref!r} hyp={hyp!r} cer={c:.3f}",flush=True)
    summary={"dataset":dataset,"n":len(data),"word_acc":round(exact/max(1,len(data)),4),"avg_cer":round(cer_sum/max(1,len(data)),4),"avg_sec":round(sec_sum/max(1,len(data)),2),"rows":rows}
    path=os.path.join(ROOT,"out",f"validate_{dataset}_{n}.json")
    with open(path,"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=1)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False,indent=1),flush=True)
    print("saved:",path,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("dataset",choices=["funsd","iiit5k"]); ap.add_argument("--n",type=int,default=20); a=ap.parse_args(); run(a.dataset,a.n)
