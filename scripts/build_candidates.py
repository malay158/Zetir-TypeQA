import argparse,sys,hashlib,re
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zetir.io import read_jsonl,write_jsonl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def key(text):return re.sub(r'\W+',' ',text.lower()).strip()

def stable_id(a,b):return 'candidate:'+hashlib.sha256((a+'|'+b).encode()).hexdigest()[:16]

def match(experts,novices,topic_threshold=.2,restrict_subject=False):
    if not experts or not novices:return []
    # This reconstruction searches over the full candidate pool; the original
    # manually retained candidate subsets and their source IDs are unavailable.
    texts=[r.get('topic') or r['question'] for r in novices]+[r['question'] for r in experts]
    vectors=TfidfVectorizer(ngram_range=(1,2),stop_words='english').fit_transform(texts)
    n=len(novices);sim=cosine_similarity(vectors[n:],vectors[:n])
    results=[];used=set()
    for i,ex in enumerate(experts):
        order=sorted(range(n),key=lambda j:(-sim[i,j],novices[j]['id']))
        for j in order:
            no=novices[j]
            if restrict_subject and ex.get('topic','') not in subject_options(no['question']):continue
            if float(sim[i,j])<topic_threshold:break
            if no['id'] in used or key(no['question'])==key(ex['question']):continue
            if not no.get('answer') or not ex.get('answer'):continue
            used.add(no['id']);results.append({'pair_id':stable_id(no['id'],ex['id']),'domain':ex['domain'],
                'topic':(no.get('topic') or ex.get('topic') or '').strip(),
                'novice_id':no['id'],'novice_query':no['question'],'novice_reference':no['answer'],
                'expert_id':ex['id'],'expert_query':ex['question'],'expert_reference':ex['answer'],
                'alignment_similarity':round(float(sim[i,j]),5),'curation_status':'pending',
                'reference_route_novice':None,'reference_route_expert':None,
                'notes':'Review topic match, information depth, and source-style confounding'});break
    return results

def subject_options(text):
    # Broad and imperfect categories. Manual review is REQUIRED for science matches.
    text=text.lower()
    groups={'biology':['cell','plant','human','blood','animal','organism','photosynth','gene','dna','body','heart','disease'],
            'chemistry':['chemical','atom','acid','reaction','molecule','element','compound'],
            'physics':['force','energy','motion','electric','resistance','speed','magnet','light'],
            'astronomy':['planet','star','solar','moon','space','sun','earth','galaxy'],
            'mathematics':['number','equation','algebra','angle','probability','geometry']}
    active={name for name,words in groups.items() if any(w in text for w in words)}
    if not active:active=set(groups)
    out=set()
    for name in active:out.update([f'high_school_{name}',f'college_{name}',name])
    out.update(['general science','conceptual_physics','anatomy','clinical_knowledge','college_medicine'])
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--output',required=True);p.add_argument('--medical-threshold',type=float,default=.2);p.add_argument('--education-threshold',type=float,default=.12)
    a=p.parse_args();src=list(read_jsonl(a.sources));by=defaultdict(list)
    for r in src:by[(r.get('domain'),r.get('type'))].append(r)
    # Use research queries as the anchor for medical and grade-school queries as anchor for education.
    medical=match(by['medical','expert'],by['medical','novice'],a.medical_threshold)
    edu=match(by['educational','expert'],by['educational','novice'],a.education_threshold,restrict_subject=False)
    out=medical+edu;write_jsonl(a.output,out)
    print(f'Candidate pairs: medical={len(medical)}, educational={len(edu)}; ALL PENDING CURATION')
if __name__=='__main__':main()
