import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zetir.io import read_jsonl,write_jsonl
from zetir.graph import read_graph,load_nlp,structural_scores
from zetir.retrieve import DenseIndex
from zetir.inference import route_and_retrieve,load_generator,generate_answer,PROMPTS
from zetir.type_model import load_ranker

def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',required=True);p.add_argument('--resources',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',default='test');p.add_argument('--generate',action='store_true')
    p.add_argument('--encoder',default='sentence-transformers/all-MiniLM-L6-v2')
    a=p.parse_args();from sentence_transformers import SentenceTransformer
    encoder=SentenceTransformer(a.encoder);nlp=load_nlp();ranker=load_ranker(a.weights)
    caches={};model=tok=None
    if a.generate:model,tok=load_generator()
    for dom in ('medical','educational'):
        root=Path(a.resources);kg_path=root/f'{dom}_kg.json';chunks=root/f'{dom}_chunks.jsonl'
        if not kg_path.exists():continue
        graph=read_graph(kg_path);nodes=sorted(graph.nodes)
        matrix=np.asarray(encoder.encode(nodes,normalize_embeddings=True,show_progress_bar=False))
        dense=DenseIndex(encoder,list(read_jsonl(chunks)))
        caches[dom]=(graph,nodes,matrix,structural_scores(graph),dense)
    out=[]
    for p in read_jsonl(a.pairs):
        if p['split']!=a.split:continue
        if p['domain'] not in caches:raise ValueError('Missing domain resources: '+p['domain'])
        graph,nodes,vectors,scores,dense=caches[p['domain']]
        for typ in ('novice','expert'):
            q=p[f'{typ}_query'];start=time.perf_counter()
            r=route_and_retrieve(q,graph,dense,nlp,encoder,ranker,nodes,vectors,scores)
            elapsed=1000*(time.perf_counter()-start)
            result={'pair_id':p['pair_id'],'query_id':p[f'{typ}_id'],'type':typ,'domain':p['domain'],
                    'predicted_route':r['route'],'reference_route':p.get(f'reference_route_{typ}'),
                    'tau':r['tau'],'fallback':r['fallback'],'evidence':r['evidence'],
                    'retrieval_and_routing_ms':elapsed,'reference_answer':p.get(f'{typ}_reference','')}
            if a.generate:
                start=time.perf_counter()
                result['answer']=generate_answer(q,r['evidence'],r['route'],model,tok)
                result['generation_ms']=1000*(time.perf_counter()-start)
            out.append(result)
    write_jsonl(a.output,out);print(f'Predictions written: {len(out)}. No score is fabricated.')
if __name__=='__main__':main()
