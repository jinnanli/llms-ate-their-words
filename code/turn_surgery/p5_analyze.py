"""Offline statistical closure for P5 Layer A, NEUTRAL_ACK, and surgery."""
from __future__ import annotations

import argparse, csv, json, math, random, statistics
from collections import defaultdict
from pathlib import Path

MODELS = ["gpt-5.6-luna", "grok-4.5", "gpt-4o", "deepseek-ai/DeepSeek-V3.2", "Qwen/Qwen3-14B"]
TASKS = ["actions", "code", "data2text", "database", "math", "summary"]
MODES = ["FULL", "CONCAT", "SHARDED"]
SEED = 20260812
BOOT = 10000


def csv_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))


def jsonl(path):
    if not Path(path).exists(): return []
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def mean(xs): return statistics.fmean(xs) if xs else None


def quantile(xs, p):
    xs=sorted(xs); x=(len(xs)-1)*p; lo=math.floor(x); hi=math.ceil(x)
    return xs[lo] if lo==hi else xs[lo]*(hi-x)+xs[hi]*(x-lo)


def paired_ci(rows, expression, n=BOOT):
    if not rows: return {"mean":None,"ci_low":None,"ci_high":None,"n":0}
    values=[expression(r) for r in rows]; rng=random.Random(SEED); draws=[]
    for _ in range(n): draws.append(mean([values[rng.randrange(len(values))] for __ in values]))
    return {"mean":mean(values),"ci_low":quantile(draws,.025),"ci_high":quantile(draws,.975),"n":len(values)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",required=True); ap.add_argument("--neutral",required=True); ap.add_argument("--surgery",required=True); ap.add_argument("--stability",required=True); ap.add_argument("--output-dir",default="."); a=ap.parse_args(); out=Path(a.output_dir)
    source=csv_rows(a.source)
    complete=[]
    for r in source:
        if all(r.get(f"{m}_score") not in (None, "") for m in MODES):
            for m in MODES:r[m]=float(r[f"{m}_score"])
            complete.append(r)
    panel=[]; gaps=[]
    for model in MODELS:
      for task in TASKS:
        cell=[r for r in complete if r['tested_model']==model and r['task']==task]
        for mode in MODES:
          ci=paired_ci(cell,lambda r,m=mode:r[m]); panel.append({'model':model,'task':task,'mode':mode,**ci})
        for name,fn in [('FULL-SHARDED',lambda r:r['FULL']-r['SHARDED']),('CONCAT-SHARDED',lambda r:r['CONCAT']-r['SHARDED']),('FULL-CONCAT',lambda r:r['FULL']-r['CONCAT'])]:
          gaps.append({'model':model,'task':task,'contrast':name,**paired_ci(cell,fn)})
    write_csv(out/'P5_MODEL_PANEL_RESULTS.csv',panel)

    neutral=[r for r in jsonl(a.neutral) if r.get('status')=='completed']
    nmap={(r['instruction_id'],r['tested_model']):r for r in neutral}
    decomposition=[]
    for model in MODELS:
      for task in TASKS:
        paired=[]
        for r in complete:
          n=nmap.get((r['instruction_id'],r['tested_model']))
          if r['tested_model']==model and r['task']==task and n:
            paired.append({**r,'NEUTRAL_ACK':float(n['final_score'])})
        specs=[('TOTAL_GAP',lambda r:r['FULL']-r['SHARDED']),('DISTRIBUTION_COMPONENT',lambda r:r['FULL']-r['NEUTRAL_ACK']),('ASSISTANT_HISTORY_COMPONENT',lambda r:r['NEUTRAL_ACK']-r['SHARDED'])]
        for name,fn in specs: decomposition.append({'model':model,'task':task,'component':name,**paired_ci(paired,fn)})
    write_csv(out/'P5_DEGRADATION_DECOMPOSITION.csv',decomposition)
    write_csv(out/'P5_NEUTRAL_ACK_RESULTS.csv',[{'trajectory_id':r['trajectory_id'],'instruction_id':r['instruction_id'],'task':r['task'],'model':r['tested_model'],'neutral_ack_score':r['final_score'],'original_sharded_score':r['original_sharded_score'],'effect':r['effect'],'status':r['status']} for r in neutral])

    surgery=[r for r in jsonl(a.surgery) if r.get('status')=='completed']
    write_csv(out/'P5_TURN_SURGERY_RESULTS.csv',[{'trajectory_id':r['trajectory_id'],'instruction_id':r['instruction_id'],'task':r.get('task'),'model':r.get('tested_model'),'surgery_turn':r['surgery_turn'],'original_score':r['original_sharded_score'],'counterfactual_score':r['final_score'],'effect':r['effect'],'fail_to_success':float(r['original_sharded_score'])<1 and float(r['final_score'])>=1,'status':r['status']} for r in surgery])
    stability=[r for r in jsonl(a.stability) if r.get('status')=='completed']
    write_csv(out/'P5_TURN_SURGERY_STABILITY.csv',[{'trajectory_id':r['trajectory_id'],'task':r.get('task'),'model':r.get('tested_model'),'surgery_turn':r['surgery_turn'],'repeat_index':r['repeat_index'],'effect':r['effect'],'final_score':r['final_score']} for r in stability])

    bytraj=defaultdict(list)
    for r in surgery:bytraj[r['trajectory_id']].append(r)
    loc=[]
    for tid,rs in bytraj.items():
        rs.sort(key=lambda x:int(x['surgery_turn'])); positive=[x for x in rs if float(x['effect'])>0]; ftp=[x for x in rs if float(x['original_sharded_score'])<1 and float(x['final_score'])>=1]; maxr=max(rs,key=lambda x:float(x['effect']))
        loc.append({'trajectory_id':tid,'task':rs[0].get('task'),'model':rs[0].get('tested_model'),'branch_count':len(rs),'earliest_positive_turn':positive[0]['surgery_turn'] if positive else None,'earliest_fail_to_success_turn':ftp[0]['surgery_turn'] if ftp else None,'maximum_impact_turn':maxr['surgery_turn'],'maximum_effect':maxr['effect'],'harmful_history_sensitive_turns':len(positive),'normalized_earliest_positive_position':(int(positive[0]['surgery_turn'])/(len(rs))) if positive else None})
    stats={'bootstrap':{'iterations':BOOT,'seed':SEED,'unit':'instruction/model paired within mode'},'source_trajectories':len(source),'complete_case_trajectories':len(complete),'neutral_completed':len(neutral),'surgery_branches_completed':len(surgery),'surgery_trajectories':len(bytraj),'fail_to_success_trajectories':sum(bool(x['earliest_fail_to_success_turn'] not in (None,'')) for x in loc),'model_panel':panel,'mode_gaps':gaps,'decomposition':decomposition,'causal_localization':loc}
    (out/'P5_FINAL_STATISTICAL_RESULTS.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps({k:stats[k] for k in ['source_trajectories','neutral_completed','surgery_branches_completed','surgery_trajectories','fail_to_success_trajectories']}))

if __name__=='__main__':main()
