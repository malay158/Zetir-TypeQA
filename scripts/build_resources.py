import sys,argparse,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zetir.io import read_jsonl,write_jsonl
from zetir.retrieve import build_chunks
from zetir.graph import build_graph,load_nlp,save_graph

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--pairs');p.add_argument('--output-dir',required=True)
    p.add_argument('--include-test-answer-leakage',action='store_true',help='NOT RECOMMENDED; diagnostic only')
    a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # External documents may be provided as source JSONL. Separate evaluation source IDs by pair provenance.
    held_out=set()
    if a.pairs and not a.include_test_answer_leakage:
        for pair in read_jsonl(a.pairs):
            if pair.get('split')=='test':held_out.update([pair['novice_id'],pair['expert_id']])
    sources=list(read_jsonl(a.sources));corpus=[]
    for row in sources:
        if row['id'] in held_out:continue
        passage=row.get('answer') or ''
        # PubMedQA contexts are provided as additional evidence.
        extra='\n'.join(row.get('contexts',[]))
        content='\n'.join([x for x in (passage,extra) if x])
        if content:corpus.append({'id':row['id'],'domain':row['domain'],'text':content})
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    nlp=load_nlp()
    for domain in ('medical','educational'):
        chunks=list(build_chunks((x for x in corpus if x['domain']==domain),tok))
        if not chunks:continue
        write_jsonl(out/f'{domain}_chunks.jsonl',chunks)
        print(domain,'chunks=',len(chunks),'building KG...')
        kg=build_graph(chunks,nlp)
        save_graph(kg,out/f'{domain}_kg.json')
        print(domain,'KG nodes=',len(kg),'edges=',kg.number_of_edges())
    print('Paper-specific chunk count is not guaranteed.')
if __name__=='__main__':main()
