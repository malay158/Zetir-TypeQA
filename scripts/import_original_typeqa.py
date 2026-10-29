import argparse,csv,sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zetir.io import write_jsonl
from scripts.audit_typeqa import audit

def import_manifest(path):
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            yield {'pair_id':r['pair_id'],'domain':r['domain'],'topic_group':r['topic_group'],
                   'novice_id':r['novice_id'],'expert_id':r['expert_id'],
                   'novice_query':r['novice_query'],'expert_query':r['expert_query'],
                   'novice_reference':r['novice_reference'],'expert_reference':r['expert_reference'],
                   'split':r['split'],'reference_route_novice':r.get('reference_route_novice') or None,
                   'reference_route_expert':r.get('reference_route_expert') or None,
                   'curation_status':'approved','record_origin':'verified_original_manifest'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--check-paper-counts',action='store_true')
    a=p.parse_args();rows=list(import_manifest(a.manifest));audit(rows,a.check_paper_counts)
    write_jsonl(a.output,rows)
    print('Original manifest imported.')
if __name__=='__main__':main()
