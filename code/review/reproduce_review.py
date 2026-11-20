"""CPU reproduction of the frozen review analyses from numeric-only tables.

Algorithms/ordering/seeds are ported from analyze_five_v2.py and final_*.py.
No model, network, developer workspace, or raw conversation dependency.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/review'
OUT=ROOT/'results/reproduced/review'

def ci(values,seed=20261119):
    v=np.asarray(values,dtype=float)
    assert np.isfinite(v).all()
    ix=np.random.default_rng(seed).integers(len(v),size=(10000,len(v)))
    lo,hi=np.quantile(v[ix].mean(1),[.025,.975])
    return dict(estimate=v.mean(),ci_low=lo,ci_high=hi)

def save(name,df):
    df.to_csv(OUT/name,index=False)
    expected=ROOT/'results/review'/name
    if expected.exists():
        ref=pd.read_csv(expected)
        assert list(df.columns)==list(ref.columns),(name,'columns')
        assert len(df)==len(ref),(name,'rows')
        for key in df:
            if pd.api.types.is_numeric_dtype(df[key]):
                assert np.allclose(df[key],ref[key],atol=1e-12,rtol=1e-10,equal_nan=True),(name,key)
            else:
                assert df[key].fillna('').astype(str).tolist()==ref[key].fillna('').astype(str).tolist(),(name,key)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cells=pd.read_csv(ROOT/'data/rq1/rq1_cell_summary.csv')
    panel=pd.read_csv(ROOT/'results/rq1/P5_MODEL_PANEL_RESULTS.csv')
    for _,r in panel.iterrows():
        c=cells[(cells.model==r.model)&(cells.task==r.task)].iloc[0]
        assert np.isclose(c['score_'+r['mode'].lower()],r['mean']) and c.paired_complete_n==r['n']
    cells.groupby('model')[['score_full','score_concat','score_sharded','degradation_full_sharded']].mean().to_csv(OUT/'rq1_task_macro_point_estimates.csv')
    b=pd.read_csv(DATA/'rq3_branches.csv')
    assert len(b)==219 and b.trajectory_id.nunique()==40
    assert not b.duplicated(['trajectory_id','target_turn']).any()
    assert np.allclose(b.S-b.N,b.delta_net)
    assert b.groupby('task').size().to_dict()==dict(actions=35,code=53,data2text=43,math=38,summary=50)
    # Frozen primary ordering is trajectory_id, not task/trajectory_id.
    tr=b.groupby('trajectory_id',sort=True).agg(task=('task','first'),delta_net=('delta_net','mean'),branch_n=('delta_net','size')).reset_index()
    rows=[]
    for task in ['FIVE_TASK_ONLY','actions','code','data2text','math','summary']:
        g=tr if task=='FIVE_TASK_ONLY' else tr[tr.task==task]
        v=g.delta_net.to_numpy();w=g.branch_n.to_numpy()
        ix=np.random.default_rng(20261118).integers(len(v),size=(10000,len(v)))
        for weighting in ['trajectory_equal','branch_weighted']:
            draws=v[ix].mean(1) if weighting=='trajectory_equal' else (v[ix]*w[ix]).sum(1)/w[ix].sum(1)
            lo,hi=np.quantile(draws,[.025,.975])
            rows.append(dict(task=task,weighting=weighting,estimate=v.mean() if weighting=='trajectory_equal' else np.average(v,weights=w),ci_low=lo,ci_high=hi,trajectory_n=len(v),branch_n=int(w.sum())))
    save('rq3_statistics.csv',pd.DataFrame(rows))
    t=b.groupby(['task','trajectory_id'],as_index=False).delta_net.mean()
    save('leave_one_task_out.csv',pd.DataFrame([dict(excluded_task=k,trajectory_n=len(g),**ci(g.delta_net)) for k,g in [('NONE',t)]+[(k,t[t.task!=k]) for k in sorted(t.task.unique())]]))
    save('leave_one_trajectory_out.csv',pd.DataFrame([dict(excluded_trajectory=k,**ci(t[t.trajectory_id!=k].delta_net)) for k in t.trajectory_id]))
    classes=[]
    for eps in [0,.01]:
        for task,g in b.groupby('task'):
            threshold=eps if task in ['data2text','summary'] else 0
            classes.append(dict(task=task,continuous_epsilon=eps,positive=int((g.delta_net>threshold).sum()),negative=int((g.delta_net < -threshold).sum()),zero=int((g.delta_net.abs()<=threshold).sum()),branches=len(g)))
    save('practical_effect_classes.csv',pd.DataFrame(classes))
    s=pd.read_csv(DATA/'stability_pairs.csv')
    assert len(s)==15 and not s.duplicated(['pair_id','repeat']).any()
    summary=s.groupby('task').agg(repeats=('repeat','size'),mean_net=('delta_net','mean'),min_net=('delta_net','min'),max_net=('delta_net','max'),sd_net=('delta_net','std'),null_unique_outputs=('N_output_hash','nunique'),surgery_unique_outputs=('S_output_hash','nunique'))
    summary['sign_consistent']=s.groupby('task').delta_net.apply(lambda x:np.sign(x).nunique()==1)
    save('stability_summary.csv',summary.reset_index())
    r=pd.read_csv(DATA/'rq2_pairs.csv').sort_values(['task','trajectory_id']).reset_index(drop=True)
    assert len(r)==40 and r.trajectory_id.nunique()==40 and (r.groupby('task').size()==8).all()
    for name,a,z in [('SHORT_minus_N','SHORT','N'),('MATCHED_minus_N','MATCHED','N'),('MATCHED_minus_SHORT','MATCHED','SHORT')]:
        assert np.allclose(r[name],r[a]-r[z])
    save('rq2_statistics.csv',pd.DataFrame([dict(task=k,metric=m,trajectory_n=len(g),**ci(g[m])) for k,g in [('five_task_selected',r)]+list(r.groupby('task')) for m in ['SHORT_minus_N','MATCHED_minus_N','MATCHED_minus_SHORT']]))
    token=pd.read_csv(DATA/'token_by_trajectory.csv')
    piv=token.pivot(index=['task','trajectory_id'],columns='condition',values='assistant_content_tokens')
    piv['relative_difference']=(piv.length_matched_neutralization-piv.null_replay)/piv.null_replay
    save('token_task_summary.csv',piv.groupby('task').agg(original_mean=('null_replay','mean'),short_mean=('short_neutralization','mean'),matched_mean=('length_matched_neutralization','mean'),relative_mean=('relative_difference','mean'),relative_min=('relative_difference','min'),relative_max=('relative_difference','max')).reset_index())
    p=token.pivot(index=['task','trajectory_id'],columns='condition',values='whole_prompt_tokens')
    p['relative_difference']=(p.length_matched_neutralization-p.null_replay)/p.null_replay
    save('token_prompt_task_summary.csv',p.groupby('task').agg(original_prompt_mean=('null_replay','mean'),short_prompt_mean=('short_neutralization','mean'),matched_prompt_mean=('length_matched_neutralization','mean'),mean_relative_difference=('relative_difference','mean')).reset_index())
    correlation()
    pd.DataFrame(rows).query("weighting=='trajectory_equal'").plot(x='task',y='estimate',kind='bar',legend=False,title='RQ3: Surgery minus Null (selected Qwen cohort)')
    plt.ylabel('Trajectory-equal score difference');plt.tight_layout();plt.savefig(OUT/'rq3_net_effect.png',dpi=180);plt.close()
    print('REVIEW_REPRODUCTION: PASS (frozen numeric checks, tolerance 1e-12 absolute)')

def correlation():
    d=pd.read_csv(DATA/'rq1_rq2_cells.csv');assert len(d)==30
    def corr(g):
        a=g.degradation_full_sharded;b=g.rq2_mean_short_delta
        return dict(spearman=a.corr(b,method='spearman'),pearson=a.corr(b))
    groups=[('all','all',d)]+[('task',k,g) for k,g in d.groupby('task')]+[('model',k,g) for k,g in d.groupby('model')]
    for dim in ['model','task']:
        groups += [('leave_out_'+dim,k,d[d[dim]!=k]) for k in sorted(d[dim].unique())]
    save('rq1_rq2_correlations.csv',pd.DataFrame([dict(group_type=t,group=k,cell_n=len(g),**corr(g)) for t,k,g in groups]))
    x=d.pivot(index='model',columns='task',values='degradation_full_sharded').to_numpy()
    y=d.pivot(index='model',columns='task',values='rq2_mean_short_delta').to_numpy()
    rng=np.random.default_rng(20261119);draws=[]
    for _ in range(10000):
        mi=rng.integers(5,size=5);ti=rng.integers(6,size=6)
        a=x[np.ix_(mi,ti)].ravel();b=y[np.ix_(mi,ti)].ravel()
        if np.std(a)==0 or np.std(b)==0:continue
        draws.append([np.corrcoef(a,b)[0,1],np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1]])
    bs=np.asarray(draws)
    save('rq1_rq2_crossed_bootstrap.csv',pd.DataFrame([dict(metric=k,ci_low=np.quantile(bs[:,i],.025),ci_high=np.quantile(bs[:,i],.975),valid_replicates=len(bs),unit='crossed model/task cell resampling; descriptive only') for i,k in enumerate(['pearson','spearman'])]))
    fig,axes=plt.subplots(2,3,figsize=(10,6),sharex=True,sharey=True)
    models=sorted(d.model.unique())
    for ax,(task,g) in zip(axes.flat,d.groupby('task')):
        for _,r in g.iterrows():ax.scatter(r.degradation_full_sharded,r.rq2_mean_short_delta,color=plt.get_cmap('tab10')(models.index(r.model)),label=r.model)
        ax.axhline(0,color='gray',lw=.6);ax.set_title(task)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,fontsize=8)
    fig.supxlabel('FULL - SHARDED');fig.supylabel('SHORT - SHARDED');fig.tight_layout(rect=(.02,.02,1,.9));fig.savefig(OUT/'rq1_rq2_relation.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
