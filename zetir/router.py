"""Three-way type-conditioned retrieval."""
from dataclasses import dataclass

@dataclass
class Router:
    lower: float = .38
    upper: float = .45
    def __post_init__(self):
        if not (0 <= self.lower < self.upper <= 1): raise ValueError('Invalid thresholds')
    def route(self, tau):
        if tau < self.lower: return 'dense'
        if tau > self.upper: return 'graph'
        return 'hybrid'
    def graph_weight(self, tau):
        return max(0., min(1., (tau-self.lower)/(self.upper-self.lower)))

def normalise_minmax(scores):
    scores=list(scores)
    if not scores:return []
    lo=min(scores); hi=max(scores)
    return [(s-lo)/(hi-lo) if hi>lo else 1.0 for s in scores]

def fuse_candidates(dense, graph, omega, top_k=5):
    """Dense and KG candidates must share stable evidence IDs; missing-channel scores = zero.
    Inputs iterable of (id, text, raw_score) for EACH channel.
    """
    ds=normalise_minmax([r[2] for r in dense]); gs=normalise_minmax([r[2] for r in graph]); items={}
    for r,s in zip(dense,ds): items[r[0]]={'id':r[0],'text':r[1],'dense':s,'graph':0.}
    for r,s in zip(graph,gs):
        x=items.setdefault(r[0],{'id':r[0],'text':r[1],'dense':0.,'graph':0.})
        x['graph']=s
    out=[]
    for x in items.values():
        x['score']=(1-omega)*x['dense']+omega*x['graph'];out.append(x)
    return sorted(out,key=lambda x:(-x['score'],x['id']))[:top_k]
