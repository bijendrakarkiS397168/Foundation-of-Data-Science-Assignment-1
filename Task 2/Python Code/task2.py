"""All 72 matches: accuracy-versus-volume slopes within twelve groups."""
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import read_source, group_mapping, compare_previous, require

ROOT=Path(__file__).resolve().parents[1]

def slope(shots,accuracy):
    """OLS slope with intercept, in percentage points per additional shot."""
    x,y=np.asarray(shots,dtype=float),np.asarray(accuracy,dtype=float)
    require(len(x)==len(y) and len(x)>=3 and np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid regression inputs.')
    centered=x-x.mean()
    require(float(centered@centered)>0, 'Shot volume has no variation; slope undefined.')
    beta=float(centered@(y-y.mean())/(centered@centered))
    require(np.isclose(beta,stats.linregress(x,y).slope), 'Slope formula check failed.')
    return beta,float(y.mean()-beta*x.mean())

def infer(values):
    x=np.asarray(values,dtype=float); n=len(x)
    require(n>=3 and np.isfinite(x).all(), 'At least three finite group slopes required.')
    mean,sd=float(x.mean()),float(x.std(ddof=1))
    require(sd>0, 'Zero group variance: t-test undefined.')
    se=sd/math.sqrt(n); t=mean/se
    p=float(2*stats.t.sf(abs(t),n-1)); margin=float(stats.t.ppf(.975,n-1)*se)
    check=stats.ttest_1samp(x,0,alternative='two-sided')
    require(np.isclose(t,check.statistic) and np.isclose(p,check.pvalue), 't-test verification failed.')
    boot=np.random.default_rng(20260906).choice(x,(20000,n),replace=True).mean(axis=1)
    signs=np.asarray(list(itertools.product([-1,1],repeat=n)))@x/n
    return dict(n=n,df=n-1,mean=mean,sd=sd,median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max()),
                se=se,t=t,p=p,ci_low=mean-margin,ci_high=mean+margin,dz=mean/sd,
                bonferroni_p_four_planned_tests=min(1,4*p),shapiro_p=float(stats.shapiro(x).pvalue),
                bootstrap_ci_low=float(np.quantile(boot,.025)),bootstrap_ci_high=float(np.quantile(boot,.975)),
                sign_flip_p=float(np.mean(np.abs(signs)>=abs(mean)-1e-12)),
                leave_one_group_out_min=float(((x.sum()-x)/(n-1)).min()),
                leave_one_group_out_max=float(((x.sum()-x)/(n-1)).max()))

def write_csv(path,rows,fields=None):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def prepare(matches):
    mapping=group_mapping(matches); observations=[]
    for r in matches:
        for side,other in [('1','2'),('2','1')]:
            shots,sot=r['shots'+side],r['sot'+side]
            require(shots>0, 'Zero shots found: accuracy undefined. Stop and revise the analysis; no rows silently excluded.')
            observations.append({'source_row':r['source_row'],'date':r['date'],'group_id':mapping[r['team'+side]],
                                 'team':r['team'+side],'opponent':r['team'+other],
                                 'shots':shots,'shots_on_target':sot,'accuracy_pct':100*sot/shots,
                                 'equal_shot_volume_match':r['shots1']==r['shots2']})
    require(len(observations)==144, 'All 144 team-match observations must be retained.')
    return observations,mapping

def run(root=ROOT):
    raw=root/'data/raw/published_world_cup.csv'; matches=read_source(raw)
    source_check=compare_previous(matches,root/'data/raw/previous_group_stage_capture.tsv')
    observations,mapping=prepare(matches); groups=[]
    for group in sorted(set(mapping.values())):
        rows=[r for r in observations if r['group_id']==group]
        require(len(rows)==12, 'Every group must retain all 12 team-match observations.')
        beta,intercept=slope([r['shots'] for r in rows],[r['accuracy_pct'] for r in rows])
        groups.append({'group_id':group,'teams':' | '.join(sorted(t for t,g in mapping.items() if g==group)),
                       'matches':6,'team_match_observations':len(rows),
                       'mean_shots':float(np.mean([r['shots'] for r in rows])),
                       'mean_accuracy_pct':float(np.mean([r['accuracy_pct'] for r in rows])),
                       'slope_pp_per_shot':beta,'intercept_pct':intercept})
    results=infer([g['slope_pp_per_shot'] for g in groups])
    results.update(included_matches=72,excluded_matches=0,team_match_observations=144,
                   unit='percentage points of accuracy per additional shot',
                   test='Two-sided one-sample t-test of 12 group slopes against zero',alpha=.05)
    for folder in ['data/processed','results','figures']: (root/folder).mkdir(parents=True,exist_ok=True)
    write_csv(root/'data/processed/group_stage_shots.csv',matches)
    write_csv(root/'results/team_match_accuracy_audit.csv',observations)
    write_csv(root/'results/group_slopes_task2.csv',groups)
    write_csv(root/'results/statistical_results_task2.csv',[results])
    ties=[r for r in matches if r['shots1']==r['shots2']]
    write_csv(root/'results/retained_equal_volume_matches.csv',ties,list(matches[0]))
    summaries=[]
    for name in ['shots','shots_on_target','accuracy_pct']:
        v=np.asarray([r[name] for r in observations])
        summaries.append(dict(variable=name,n=len(v),mean=float(v.mean()),sd=float(v.std(ddof=1)),median=float(np.median(v)),minimum=float(v.min()),maximum=float(v.max())))
    write_csv(root/'results/descriptive_statistics.csv',summaries)
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
              'source_comparison':source_check,'python':platform.python_version(),
              'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib']},'bootstrap_seed':20260906,
              'data_status':'Direct published CSV agrees with previous capture on all 288 shot fields; not independent event adjudication.',
              'analysis_change':'User requested no exclusions. Replaced higher-minus-lower comparison with group slopes using every team-match.',
              'inference_scope':'Model-based comparable group slopes; not sampling uncertainty for a finite tournament census.',
              'results':results}
    (root/'results/statistical_results_task2.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    plot(root,groups,results,observations); report(root,results,ties)
    print('Task 2 completed: 72 matches retained, 0 excluded; 144 team-match observations; 12 group slopes.')
    print(f"Mean slope: {results['mean']:.4f} pp/shot; 95% CI [{results['ci_low']:.4f}, {results['ci_high']:.4f}].")
    print(f"t({results['df']})={results['t']:.4f}; p={results['p']:.6f}. See results/FINDINGS_task2.md.")
    return results

def plot(root,groups,r,observations):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    x=[g['slope_pp_per_shot'] for g in groups]; fig,ax=plt.subplots(figsize=(9,5.5))
    ax.scatter(x,range(len(x)),s=55,color='#007f86',zorder=3); ax.axvline(0,color='gray',linestyle='--')
    ax.errorbar(r['mean'],-1,xerr=[[r['mean']-r['ci_low']],[r['ci_high']-r['mean']]],fmt='D',color='#bd571c',capsize=5)
    ax.set_yticks([-1]+list(range(len(x))),['Mean (95% CI)']+[g['group_id'] for g in groups])
    ax.set_xlabel('Accuracy slope (percentage points per additional shot)')
    ax.set_title('Shot volume and accuracy — all 72 group-stage matches',loc='left',pad=12)
    fig.tight_layout(); fig.savefig(root/'figures/task2.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    stats.probplot(x,dist='norm',plot=axes[0]); axes[0].set_title('Q–Q plot: 12 group slopes')
    for g in groups:
        rows=[o for o in observations if o['group_id']==g['group_id']]
        axes[1].scatter([o['shots'] for o in rows],[o['accuracy_pct'] for o in rows],s=18,alpha=.65)
    axes[1].set(xlabel='Shots per team per match',ylabel='Shots on target / shots (%)',title='All 144 team-match observations')
    fig.tight_layout(); fig.savefig(root/'figures/task2_diagnostics.png',dpi=180); plt.close(fig)

def report(root,r,ties):
    lines=['# Bijendra — Objective 1, Task 2','',
           'Question: Within World Cup 2026 groups, is shot volume associated with shot accuracy?','',
           '**All 72 matches retained; no exclusions.** Each side contributes one observation: 144 team-match observations. Accuracy = 100 × shots on target / shots.','',
           '## Retained equal-volume matches','']
    for row in ties:
        lines.append(f"- Source row {row['source_row']}: {row['team1']}–{row['team2']}, shots {row['shots1']}–{row['shots2']}, shots on target {row['sot1']}–{row['sot2']}. Both sides retain their actual accuracy.")
    lines+=['','## Results','','| Statistic | Value |','|---|---|']
    for name,key in [('Groups','n'),('Mean slope (pp per shot)','mean'),('SD of slopes','sd'),('Median slope','median'),('Minimum slope','minimum'),('Maximum slope','maximum'),('SE','se'),('t statistic','t'),('df','df'),('Two-sided p','p'),('95% CI lower','ci_low'),('95% CI upper','ci_high'),('Standardised group slope dz','dz'),('Bonferroni p for four planned tests','bonferroni_p_four_planned_tests')]:
        lines.append(f'| {name} | {r[key]:.6g} |')
    conclusion='does not establish a nonzero average group slope' if r['p']>=.05 else 'provides evidence of a nonzero average group slope under the stated assumptions'
    lines+=['',f"Mean slope: {r['mean']:.3f} percentage points of accuracy per additional shot; 95% t interval [{r['ci_low']:.3f}, {r['ci_high']:.3f}]. At alpha 0.05, the test {conclusion}. This is an association, not a causal effect. Failure to reject zero does not prove no relationship.",
            '', '## Why one sample of slopes?', '',
            'Fit accuracy = intercept + slope × shots separately within each group using all 12 team-match observations. Each group contributes one slope. Test the mean of those twelve slopes against zero: H0 mean slope = 0; H1 mean slope is not 0. Keeping repeated appearances of each team in one group avoids treating all 144 records as independent. No within-group regression p-values are used.',
            '', 'This replaces the previous higher-minus-lower accuracy differences. Equal shot volumes have no higher-volume side but are valid observations for a slope. They are not assigned zero accuracy, imputed or deleted. All groups receive equal weight; all observations receive equal weight within each group.',
            '', 'All 72 group-stage matches constitute a census of the source frame, not a random sample. The observed mean slope is descriptive. Inference requires a model of comparable groups with approximately independent, exchangeable slopes. Seeded draws and common tournament conditions limit this assumption. G01–G12 are local group IDs, not official letters.',
            '', '## Diagnostics and limitations', '',
            f"Shapiro–Wilk p={r['shapiro_p']:.4f}; n=12 is too small to establish normality. Inspect figures/task2_diagnostics.png. Seeded percentile bootstrap CI: [{r['bootstrap_ci_low']:.3f}, {r['bootstrap_ci_high']:.3f}] pp/shot. Group sign-flip p={r['sign_flip_p']:.4f}, requiring symmetry under the null. Leave-one-group-out means range from {r['leave_one_group_out_min']:.3f} to {r['leave_one_group_out_max']:.3f} pp/shot.",
            '', 'Bonferroni for the four planned tasks is supplementary. Holm requires the other tasks and is not calculated here. The 95% CI is an individual interval. This revised method was chosen after earlier results were available to satisfy the no-exclusions request: treat it as revised/exploratory, not preregistered confirmatory inference.',
            '', 'All 288 shot fields match the previous capture. This verifies extraction consistency, not every underlying event or provider definition. Accuracy is bounded and depends on shot denominators; small counts yield unstable percentages. Team strength, opponents and game state can affect volume and accuracy. Each group slope is noisy with only 12 observations. Do not extrapolate the lines beyond the observed range.']
    (root/'results/FINDINGS_task2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
