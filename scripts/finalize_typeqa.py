import argparse,sys,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zetir.io import read_jsonl,write_jsonl

def finalize(candidates,decision_path):
    with open(decision_path,newline='',encoding='utf-8') as f: decisions={r['pair_id']:r for r in csv.DictReader(f)}
    approved=[];ids=set();qids=set()
    for item in candidates:
        d=decisions.get(item['pair_id'],{})
        if d.get('decision','').lower()!='approve':continue
        if item['pair_id'] in ids or item['novice_id'] in qids or item['expert_id'] in qids:
            raise ValueError('Reused pair or source ID; inspect manual decisions')
        ids.add(item['pair_id']);qids.update([item['novice_id'],item['expert_id']])
        obj=dict(item);obj['curation_status']='approved';obj['reviewer_id']=d.get('reviewer_id','')
        obj['review_note']=d.get('note','');obj['reference_route_novice']=d.get('reference_route_novice') or None
        obj['reference_route_expert']=d.get('reference_route_expert') or None
        approved.append(obj)
    return approved

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--decisions',required=True);p.add_argument('--output',required=True);p.add_argument('--expect-original-counts',action='store_true')
    a=p.parse_args();rows=finalize(list(read_jsonl(a.candidates)),a.decisions)
    med=sum(x['domain']=='medical' for x in rows);edu=sum(x['domain']=='educational' for x in rows)
    if a.expect_original_counts and (med,edu)!=(1000,590):raise SystemExit(f'NOT original counts: medical={med}, educational={edu}')
    write_jsonl(a.output,rows);print('Approved pairs:',len(rows),'medical:',med,'educational:',edu)
    print('.')
if __name__=='__main__':main()
