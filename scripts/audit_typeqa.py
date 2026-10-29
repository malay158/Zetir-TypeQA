import argparse,sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zetir.io import read_jsonl

def audit(rows,expect_counts=False):
    rows=list(rows);errs=[];source_ids=[];groups={};pairs=set()
    for r in rows:
        for fld in ('pair_id','domain','novice_id','expert_id','novice_query','expert_query','split'):
            if not r.get(fld):errs.append(f'{r.get("pair_id","unknown")}: missing {fld}')
        pairs.add(r.get('pair_id'));source_ids +=[r.get('novice_id'),r.get('expert_id')]
        key=r.get('topic_group');split=r.get('split')
        if key and key in groups and groups[key]!=split:errs.append(f'topic group crosses splits: {key}')
        elif key:groups[key]=split
        if r.get('curation_status')!='approved':errs.append(f'pair not reviewed: {r.get("pair_id")}')
    if len(pairs)!=len(rows):errs.append('duplicate pair_id')
    if len(set(source_ids))!=len(source_ids):errs.append('source question reused between pairs')
    counts=collections.Counter((r.get('domain'),r.get('split')) for r in rows)
    if expect_counts:
        expected={('medical','train'):700,('medical','validation'):100,('medical','test'):200,
                  ('educational','train'):413,('educational','validation'):59,('educational','test'):118}
        for k,v in expected.items():
            if counts[k]!=v:errs.append(f'{k}: expected {v}; found {counts[k]}')
    print('pairs:',len(rows),'counts:',dict(counts))
    print('errors:',len(errs))
    for e in errs[:50]:print('ERROR:',e)
    if errs:raise SystemExit(1)
    return counts
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--expect-paper-counts',action='store_true');a=p.parse_args()
    audit(read_jsonl(a.input),a.expect_paper_counts)
