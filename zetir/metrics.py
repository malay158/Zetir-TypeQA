import numpy as np

def pairwise_accuracy(pairs):
    """Pairs: rows with novice_tau and expert_tau; ties count as failures."""
    pairs=list(pairs)
    return float(np.mean([p['expert_tau']>p['novice_tau'] for p in pairs])) if pairs else float('nan')

def route_accuracy(rows):
    """Route labels must be independently assigned, not computed from prediction thresholds."""
    rows=[r for r in rows if r.get('reference_route') in ('dense','hybrid','graph')]
    return float(np.mean([r['predicted_route']==r['reference_route'] for r in rows])) if rows else float('nan')

def answer_divergence(embeddings_n, embeddings_e):
    a=np.asarray(embeddings_n, dtype=float); b=np.asarray(embeddings_e, dtype=float)
    if a.shape!=b.shape or a.ndim!=2: raise ValueError('Pairwise aligned 2D embeddings required')
    denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    if np.any(denom<=0): raise ValueError('Zero embedding')
    return 1.0-np.sum(a*b,axis=1)/denom

def ranked_evidence_metrics(ranked_ids, relevance, k=5):
    """relevance: item id -> grade 0..3; requires independently judged relevance."""
    if relevance is None: raise ValueError('Gold relevance judgments are required')
    chosen=list(ranked_ids)[:k]
    grades=[float(relevance.get(key,0)) for key in chosen]
    relevant_total=sum(float(v)>0 for v in relevance.values())
    hits=sum(g>0 for g in grades)
    p=hits/k; r=hits/relevant_total if relevant_total else float('nan')
    rr=next((1/(i+1) for i,g in enumerate(grades) if g>0),0.)
    def dcg(x): return sum((2**g-1)/np.log2(i+2) for i,g in enumerate(x))
    ideal=sorted(map(float,relevance.values()),reverse=True)[:k]
    return dict(precision_at_k=p, recall_at_k=r, reciprocal_rank=rr, ndcg_at_k=(dcg(grades)/dcg(ideal) if dcg(ideal) else float('nan')))

def paired_bootstrap(values_a, values_b, seed=42, repeats=2000, confidence=.95):
    """Resample paired evaluation units; for ADS, units must be pairs or topic clusters."""
    a=np.asarray(values_a,dtype=float); b=np.asarray(values_b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a): raise ValueError('Equal nonempty paired score arrays needed')
    rng=np.random.default_rng(seed); delta=a-b
    draws=np.array([np.mean(delta[rng.integers(0,len(delta),len(delta))]) for _ in range(repeats)])
    edge=(1-confidence)/2
    return {'mean_difference':float(np.mean(delta)), 'ci_low':float(np.quantile(draws,edge)), 'ci_high':float(np.quantile(draws,1-edge))}
