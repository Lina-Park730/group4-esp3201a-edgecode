"""Computed experiment tables, figures and deterministic context controls."""
from functools import lru_cache
from pathlib import Path
from statistics import mean
import re


@lru_cache(maxsize=4)
def tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(path).expanduser().resolve()))


def context_tokens(text, options):
    """Count with a supplied local tokenizer, or report a conservative byte proxy."""
    if options.get('tokenizer_path'):
        return len(tokenizer(options['tokenizer_path']).encode(text, add_special_tokens=False).ids)
    return len(text.encode('utf-8'))


def budget_context(ranking, options):
    from .generation import prepare_context
    chosen=[]
    for result in ranking:
        text,_=prepare_context(chosen+[result],10**9)
        if context_tokens(text,options)<=options['context_token_budget']:
            chosen.append(result)
    return prepare_context(chosen,10**9)


def select_k(rows, rules):
    """Require near-best recall for every method, then apply resource constraints."""
    methods=sorted({r['method'] for r in rows});ks=sorted({r['k'] for r in rows})
    best_recall={m:max(r['recall_at_k'] for r in rows if r['method']==m) for m in methods}
    quality=[k for k in ks if all(next(r for r in rows if r['method']==m and r['k']==k)['recall_at_k']
             >= best_recall[m]-rules['recall_tolerance']-1e-12 for m in methods)]
    if not quality:raise ValueError('No common K satisfies the recall tolerance.')
    precision={k:mean(r['precision_at_k'] for r in rows if r['k']==k) for k in quality}
    best_precision=max(precision.values())
    acceptable=[k for k in quality if precision[k]>=best_precision-rules['precision_tolerance']-1e-12]
    fastest={m:min(r['retrieval_seconds'] for r in rows if r['method']==m) for m in methods}
    feasible=[k for k in acceptable if all(r['context_tokens']<=rules['max_context_tokens'] and
        r['retrieval_seconds']<=fastest[r['method']]*rules['latency_multiplier']+rules['latency_slack_seconds']
        for r in rows if r['k']==k)]
    if not feasible:
        raise ValueError('No K meets the development recall/precision and resource constraints. Review development tables and configure tolerances; evaluation remains blocked.')
    k=min(feasible)
    return k,dict(selected_k=k,rule='Smallest K within per-method recall tolerance, pooled precision tolerance and per-method context/latency limits.',
        rules=rules,best_recall=best_recall,recall_eligible=quality,precision_eligible=acceptable,resource_eligible=feasible,
        context_count_note='Local tokenizer if supplied; otherwise UTF-8 byte proxy, not measured Ollama tokens.')


def table(path,rows):
    fields=list(rows[0]);lines=['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']
    for row in rows:
        lines.append('| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k],float) else str(row[k]) for k in fields)+' |')
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_lines(path,rows,x,y,title,xlabel):
    plt=pyplot();fig,ax=plt.subplots(figsize=(8,4.5))
    for method in dict.fromkeys(r['method'] for r in rows):
        subset=sorted((r for r in rows if r['method']==method),key=lambda r:r[x])
        ax.plot([r[x] for r in subset],[r[y] for r in subset],marker='o',label=method)
    ax.set(title=title,xlabel=xlabel,ylabel=y);ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(str(path)+'.'+ext,dpi=160)
    plt.close(fig)


def plot_comparison(folder,summary,categories):
    plt=pyplot()
    for metric in ('mrr_at_k','recall_at_k','precision_at_k','mrr_at_5','recall_at_5','precision_at_5'):
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.bar([r['method'] for r in summary],[r[metric] for r in summary]);ax.set(ylabel=metric,ylim=(0,1),title='Frozen evaluation')
        fig.tight_layout();fig.savefig(folder/(metric+'.png'),dpi=160);fig.savefig(folder/(metric+'.svg'));plt.close(fig)
    cats=list(dict.fromkeys(r['category'] for r in categories));methods=list(dict.fromkeys(r['method'] for r in categories))
    for metric in ('mrr_at_k','recall_at_k','precision_at_k'):
        fig,ax=plt.subplots(figsize=(11,5));width=.24
        for i,m in enumerate(methods):
            ax.bar([x+(i-1)*width for x in range(len(cats))],
                [next(r[metric] for r in categories if r['category']==c and r['method']==m) for c in cats],width,label=m)
        ax.set_xticks(range(len(cats)),[c.replace('_','\n') for c in cats]);ax.set(ylabel=metric,ylim=(0,1));ax.legend()
        fig.tight_layout();fig.savefig(folder/('categories_'+metric+'.png'),dpi=160);fig.savefig(folder/('categories_'+metric+'.svg'));plt.close(fig)
