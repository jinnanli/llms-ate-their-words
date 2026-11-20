"""Opt-in external-data runner using the frozen replay constructors/evaluators.

Release adapter: dry-run needs only Python. Live/API execution is not certified
by the release tests and cannot reconstruct the omitted historical cohort.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from p0_common import content_hash, user_sequence_hash, replay_after_target, final_only_history

ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--mode',choices=['p0a','p0b'],required=True)
    ap.add_argument('--dataset',type=Path,default=ROOT/'data/sharded_instructions_600.json')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--live',action='store_true',help='Actually call models; otherwise input-only QA')
    args=ap.parse_args()
    rows=[json.loads(s) for s in args.manifest.read_text(encoding='utf8').splitlines() if s.strip()]
    assert len({r['pair_id'] for r in rows})==len(rows)
    names=['null_replay','surgery'] if args.mode=='p0a' else ['null_replay','short_neutralization','length_matched_neutralization']
    for row in rows:
        assert row['user_sequence_hash']==user_sequence_hash(row['original_messages'])
        for name in names:
            def fake(h,m):return dict(message='DRY_RUN',returned_model=m,response_id='not-live')
            if args.mode=='p0a':replay_after_target(row['original_messages'],row['model'],int(row['target_turn']),name=='surgery',fake)
            else:final_only_history(row['original_messages'],row['model'],name,fake)
    if not args.live:
        print(f'INPUT_QA: PASS; {len(rows)} rows; no API calls');return
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('Run live experiments from the repository root')
    sys.path[:0]=[str(ROOT/'code/evaluation'),str(ROOT/'code/interventions')]
    from p5_counterfactual_runner import call_assistant, evaluate
    from tasks.summary import eval_summhay
    samples={r['task_id']:r for r in json.loads(args.dataset.read_text(encoding='utf8'))}
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    fd=os.open(out/'worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    def store(path,value):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8');tmp.replace(path)
    os.environ['P1_CANONICAL_LOG']=str(out/'requests.jsonl')
    os.environ['P1_MAX_ATTEMPTS']='2'
    os.environ['P1_SUMMARY_EVALUATOR_MODEL']='gpt-4o-2024-08-06'
    def cached(call,args_,kwargs,key,expected=None):
        p=out/(key+'.response.json')
        if p.exists():v=json.loads(p.read_text(encoding='utf8'))
        else:
            time.sleep(3);v=call(*args_,**kwargs);store(p,v)
        if not v.get('returned_model') or not v.get('response_id'):raise ValueError('Missing response identity')
        if expected and v['returned_model']!=expected:raise ValueError('Fixed scorer returned a different model')
        return v
    raw_judge=eval_summhay.generate_json
    def judge(*a,**kw):
        assert kw.get('model')=='gpt-4o-2024-08-06'
        return cached(raw_judge,a,kw,'judge-'+content_hash([a,kw]),'gpt-4o-2024-08-06')
    eval_summhay.generate_json=judge
    try:
        for row in rows:
            conditions={}
            order=list(reversed(names)) if int(__import__('hashlib').sha256((row['pair_id']+'|0').encode()).hexdigest(),16)%2 else names
            for name in order:
                key=content_hash([row,name,args.mode,'gpt-4o-2024-08-06'])
                path=out/(key+'.condition.json')
                context=dict(experiment_id='public-null-replay',run_id=row['pair_id']+'|'+name+'|0',case_id=row['source_instance_id'],task=row['task'],mode=name,repeat_index=0,assistant_model=row['model'],call_role='ASSISTANT_UNDER_TEST')
                os.environ['P1_RUN_CONTEXT']=json.dumps(context)
                if path.exists():v=json.loads(path.read_text(encoding='utf8'))
                else:
                    def call(h,m):return cached(call_assistant,(h,m),{},'assistant-'+content_hash([context,h,m,0,1000]))
                    if args.mode=='p0a':h,g=replay_after_target(row['original_messages'],row['model'],int(row['target_turn']),name=='surgery',call)
                    else:h,g,_=final_only_history(row['original_messages'],row['model'],name,call)
                    answer,evaluator,score=evaluate(h,row['task'],row['model'],samples[row['source_instance_id']])
                    assert math.isfinite(score)
                    v=dict(input_hash=key,history=h,generated=g,score=score,answer=answer,evaluator=evaluator)
                    store(path,v)
                assert v['input_hash']==key and math.isfinite(v['score'])
                conditions[name]=v
            ids={g['returned_model'] for c in conditions.values() for g in c['generated']}
            assert len(ids)==1,'Returned generator identities differ; no immutable snapshot guarantee from alias equality'
            store(out/(content_hash(row['pair_id'])+'.pair.json'),dict(pair_id=row['pair_id'],trajectory_id=row['trajectory_id'],task=row['task'],conditions=conditions,scope='user-provided cohort; not automatically the frozen review population'))
    except Exception as e:
        with (out/'errors.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(type=type(e).__name__,error=str(e)))+'\n')
        raise
    finally:(out/'worker.lock').unlink()

if __name__=='__main__':main()
