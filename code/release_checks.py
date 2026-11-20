"""Run only documented CPU commands; sanitize paths in captured logs."""
from pathlib import Path
import subprocess
import sys
import json
import importlib.metadata
import math
ROOT=Path(__file__).resolve().parents[1]
commands=[['code/validate_package.py'],['code/review/test_replay.py'],['code/review/run_null_replay.py','--help'],['code/statistics/analyze_phase_l1.py'],['code/statistics/analyze_phase_l3.py'],['code/plotting/plot_figure1_framework.py'],['code/plotting/plot_refined_figure2.py'],['code/plotting/plot_refined_figures_3_5.py'],['code/review/reproduce_review.py']]
def main():
    logs=[]
    for cmd in commands:
        r=subprocess.run([sys.executable,*cmd],cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=1200)
        def clean(s):return s.replace(str(ROOT),'<REPOSITORY>').replace(str(Path(sys.executable).parents[1]),'<ENVIRONMENT>')
        logs.append(dict(command='python '+' '.join(cmd),exit_code=r.returncode,stdout=clean(r.stdout),stderr=clean(r.stderr)))
        print(cmd[0],r.returncode,flush=True)
    syntax=[]
    for p in (ROOT/'code').rglob('*.py'):
        compile(p.read_text(encoding='utf-8-sig'),p.relative_to(ROOT).as_posix(),'exec');syntax.append(p.relative_to(ROOT).as_posix())
    numeric=[]
    for name,rq in [('phase_l1','rq3'),('phase_l3','rq2')]:
        ref=json.loads((ROOT/'results'/rq/(name+'_results.json')).read_text(encoding='utf-8-sig'))
        new=json.loads((ROOT/'results/reproduced'/name/(name+'_results.json')).read_text(encoding='utf-8-sig'))
        checked=[]
        def walk(a,b,path=''):
            if isinstance(a,dict):
                for k in a:
                    assert k in b,(name,path,k)
                    walk(a[k],b[k],path+'/'+k)
            elif isinstance(a,list):
                assert len(a)==len(b)
                for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
            elif isinstance(a,(float,int)) and not isinstance(a,bool):
                assert math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-6) or (math.isnan(a) and math.isnan(b)),(name,path,a,b)
                checked.append(path)
        walk(ref,new);numeric.append(dict(analysis=name,numeric_values=len(checked),passed=True))
    (ROOT/'docs/release_validation.json').write_text(json.dumps(dict(python=sys.version.split()[0],packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','statsmodels','matplotlib','tabulate']},commands=logs,numeric_reference_checks=numeric,syntax_checked=syntax,api_calls=0,gpu_runs=0,passed=all(r['exit_code']==0 for r in logs)),indent=2),encoding='utf8')
    if any(r['exit_code'] for r in logs):raise SystemExit(1)
if __name__=='__main__':main()
