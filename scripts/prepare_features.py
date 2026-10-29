import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zetir.io import read_jsonl,write_jsonl
from zetir.graph import read_graph,load_nlp,structural_scores,type_inputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',required=True);p.add_argument('--resources',required=True);p.add_argument('--output',required=True);p.add_argument('--encoder',default='sentence-transformers/all-MiniLM-L6-v2')
    a=p.parse_args();from sentence_transformers import SentenceTransformer
    nlp=load_nlp();encoder=SentenceTransformer(a.encoder);prepared={}
    for d in ('medical','educational'):
        gp=Path(a.resources)/f'{d}_kg.json'
        if not gp.exists():continue
        kg=read_graph(gp);names=sorted(kg.nodes)
        vec=np.asarray(encoder.encode(names,normalize_embeddings=True,show_progress_bar=True))
        prepared[d]=(kg,names,vec,structural_scores(kg))
    out=[]
    for pair in read_jsonl(a.pairs):
        if pair.get('curation_status')!='approved':raise ValueError('Candidate pairs must be curated first')
        kg,names,vectors,scores=prepared[pair['domain']]
        n,_=type_inputs(pair['novice_query'],kg,nlp,encoder,names,vectors,scores)
        e,_=type_inputs(pair['expert_query'],kg,nlp,encoder,names,vectors,scores)
        out.append({'pair_id':pair['pair_id'],'split':pair['split'],'domain':pair['domain'],'topic_group':pair.get('topic_group'),
                    'novice':n,'expert':e})
    write_jsonl(a.output,out)
    print('Written',len(out),'paired two-input feature records.')
if __name__=='__main__':main()
