import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zetir.io import sha256_file
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root)
    result=[{'path':str(f.relative_to(root)),'size_bytes':f.stat().st_size,'sha256':sha256_file(f)} for f in sorted(root.rglob('*')) if f.is_file() and f.resolve()!=Path(a.output).resolve()]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('Hashed',len(result),'files')
