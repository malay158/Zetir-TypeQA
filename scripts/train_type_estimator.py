import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zetir.io import read_jsonl
from zetir.type_model import train_ranker

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();rows=list(read_jsonl(a.features))
    group_split={}
    for row in rows:
        k=row.get('topic_group')
        if k:
            if k in group_split and group_split[k]!=row['split']:raise ValueError(f'Topic leaked: {k}')
            group_split[k]=row['split']
    train=[r for r in rows if r['split']=='train'];valid=[r for r in rows if r['split']=='validation']
    if not train or not valid:raise SystemExit('Need train and validation pairs')
    result=train_ranker(train,valid,a.output,seed=a.seed)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
