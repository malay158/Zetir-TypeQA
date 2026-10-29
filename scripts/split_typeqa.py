import argparse,sys,re,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zetir.io import read_jsonl,write_jsonl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DSU:
    def __init__(self,n):self.p=list(range(n))
    def find(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def union(self,i,j):self.p[self.find(i)]=self.find(j)

def grouped_split(records,similarity=.80,seed=42):
    rows=list(records)
    if not rows:return rows
    if any(r.get('curation_status')!='approved' for r in rows):raise ValueError('Only approved pairs can be split')
    dsu=DSU(len(rows))
    for domain in sorted(set(r['domain'] for r in rows)):
        ids=[i for i,r in enumerate(rows) if r['domain']==domain]
        texts=[r.get('topic','')+' '+r['novice_query']+' '+r['expert_query'] for r in (rows[i] for i in ids)]
        x=TfidfVectorizer(stop_words='english',ngram_range=(1,2)).fit_transform(texts)
        sims=cosine_similarity(x)
        for a in range(len(ids)):
            for b in range(a+1,len(ids)):
                if sims[a,b]>=similarity:dsu.union(ids[a],ids[b])
        # Exact equal topic strings also group even if tf-idf similarity is low.
        bytopic={}
        for i in ids:
            t=re.sub(r'\s+',' ',rows[i].get('topic','').lower().strip())
            if t and t!='general science':
                if t in bytopic:dsu.union(i,bytopic[t])
                else:bytopic[t]=i
    groups={}
    for i in range(len(rows)):groups.setdefault(dsu.find(i),[]).append(i)
    rng=random.Random(seed);by_domain={}
    for members in groups.values():by_domain.setdefault(rows[members[0]]['domain'],[]).append(members)
    for dom,g in by_domain.items():
        rng.shuffle(g);g.sort(key=lambda chunk:-len(chunk))
        total=sum(len(chunk) for chunk in g);targets={'train':total*.7,'validation':total*.1,'test':total*.2};assigned={k:0 for k in targets}
        for members in g:
            order=sorted(targets,key=lambda split:(assigned[split]/max(1,targets[split]),['train','validation','test'].index(split)))
            split=order[0]
            assigned[split]+=len(members)
            for i in members:rows[i]['split']=split;rows[i]['topic_group']=f'{dom}-{dsu.find(i)}'
        print('GROUP-BASED',dom,assigned,'targets',targets)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--topic-similarity',type=float,default=.80)
    a=p.parse_args();write_jsonl(a.output,grouped_split(read_jsonl(a.input),a.topic_similarity,a.seed))
if __name__=='__main__':main()
