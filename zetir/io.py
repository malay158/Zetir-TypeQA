from pathlib import Path
import json, hashlib

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for i, line in enumerate(handle, 1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as exc: raise ValueError(f'{path}:{i}: {exc}') from exc

def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as h:
        for row in rows: h.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
