import argparse,sys,collections,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zetir.io import read_jsonl
from zetir.metrics import pairwise_accuracy,route_accuracy,answer_divergence,paired_bootstrap

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--baseline');p.add_argument('--output',required=True)
    a=p.parse_args();rows=list(read_jsonl(a.predictions));by=collections.defaultdict(dict)
    for r in rows:by[r['pair_id']][r['type']]=r
    pairs=[{'novice_tau':v['novice']['tau'],'expert_tau':v['expert']['tau']} for v in by.values() if 'novice' in v and 'expert' in v]
    result={'n_queries':len(rows),'n_complete_pairs':len(pairs),'pairwise_type_accuracy':(None if np.isnan(pairwise_accuracy(pairs)) else pairwise_accuracy(pairs)),
            'route_accuracy':(None if np.isnan(route_accuracy(rows)) else route_accuracy(rows)),'route_evaluable_n':sum(r.get('reference_route') in ('dense','hybrid','graph') for r in rows)}
    if rows and all('answer' in r for r in rows):
        from rouge_score import rouge_scorer
        from bert_score import score as bertscore
        scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
        hyps=[r['answer'] for r in rows];refs=[r['reference_answer'] for r in rows]
        if any(not ref for ref in refs):raise ValueError('Some source reference answers are missing; cannot report quality metrics')
        rouge=[scorer.score(ref,hyp)['rougeL'].fmeasure for ref,hyp in zip(refs,hyps)]
        _,_,bs=bertscore(hyps,refs,lang='en',verbose=False)
        result.update({'rougeL_mean':float(np.mean(rouge)),'bertscore_F1_mean':float(bs.mean())})
        from sentence_transformers import SentenceTransformer
        encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        aligned=[v for v in by.values() if 'novice' in v and 'expert' in v]
        ne=encoder.encode([v['novice']['answer'] for v in aligned]);ee=encoder.encode([v['expert']['answer'] for v in aligned])
        result['answer_divergence_mean']=float(np.mean(answer_divergence(ne,ee)))
    if a.baseline:
        other=list(read_jsonl(a.baseline));index={r['query_id']:r for r in other}
        ids=[r['query_id'] for r in rows if r['query_id'] in index]
        if len(ids)!=len(rows):raise ValueError('Baseline and method evaluation queries must match')
        if all('rougeL' in r for r in rows) and all('rougeL' in index[x] for x in ids):
            result['paired_rougeL_difference']=paired_bootstrap([r['rougeL'] for r in rows],[index[r['query_id']]['rougeL'] for r in rows])
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
