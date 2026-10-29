from pathlib import Path
import json
import numpy as np
from .io import read_jsonl, write_jsonl

def chunk_by_token_ids(text,tokenizer,size=256,overlap=50):
    if size<=overlap:raise ValueError('chunk size must exceed overlap')
    ids=tokenizer.encode(text,add_special_tokens=False)
    for start in range(0,len(ids),size-overlap):
        part=ids[start:start+size]
        if part:yield tokenizer.decode(part,skip_special_tokens=True)
        if start+size>=len(ids):break

def build_chunks(source_rows,tokenizer):
    for row in source_rows:
        for i,text in enumerate(chunk_by_token_ids(row['text'],tokenizer)):
            yield {'id':f"{row['id']}:chunk{i}",'domain':row['domain'],'source_id':row['id'],'text':text}

class DenseIndex:
    def __init__(self,encoder,records):
        self.encoder=encoder;self.records=list(records)
        if not self.records:raise ValueError('No retrieval documents')
        self.vectors=np.asarray(encoder.encode([r['text'] for r in self.records],normalize_embeddings=True,show_progress_bar=False),dtype='float32')
        try:
            import faiss
            self.index=faiss.IndexFlatIP(self.vectors.shape[1]);self.index.add(self.vectors)
        except ImportError:self.index=None
    def search(self,query,k=5):
        q=np.asarray(self.encoder.encode([query],normalize_embeddings=True),dtype='float32')
        if self.index is not None:
            scores,indices=self.index.search(q,min(k,len(self.records)))
            pairs=zip(indices[0],scores[0])
        else:
            sims=self.vectors@q[0];indices=np.argsort(-sims)[:k];pairs=((i,sims[i]) for i in indices)
        return [(self.records[int(i)]['id'],self.records[int(i)]['text'],float(score)) for i,score in pairs]
