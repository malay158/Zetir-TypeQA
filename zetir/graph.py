import json, math
from collections import Counter
from pathlib import Path
import networkx as nx
import numpy as np

def load_nlp(model='en_core_web_sm'):
    import spacy
    try: return spacy.load(model)
    except OSError as exc: raise RuntimeError('Run: python -m spacy download en_core_web_sm') from exc

def concepts(doc, limit=18):
    chunks=[x.text for x in doc.ents]
    if doc.has_annotation('DEP'): chunks += [x.text for x in doc.noun_chunks]
    seen=set();result=[]
    for term in chunks:
        txt=' '.join(t.lemma_.lower() for t in doc.vocab.make_doc(term) if not t.is_stop and t.is_alpha)
        # make_doc() does not run the lemmatiser; fall back to surface form here.
        if not txt: txt=' '.join(t.text.lower() for t in doc.vocab.make_doc(term) if t.is_alpha)
        if len(txt)>=3 and txt not in seen: seen.add(txt);result.append(txt)
        if len(result)>=limit:break
    return result

def build_graph(records, nlp, max_concepts=18):
    """Input records with `text` and stable id; normalised co-occurrence edge weights."""
    graph=nx.Graph(); freq=Counter();pair_counts=Counter();corpus_count=0
    for row in records:
        text=row.get('text','').strip()
        if not text:continue
        corpus_count+=1
        detected=concepts(nlp(text),max_concepts)
        freq.update(detected)
        for i,a in enumerate(detected):
            for b in detected[i+1:]:pair_counts[tuple(sorted((a,b)))]+=1
    for name,count in freq.items():graph.add_node(name, frequency=count)
    for (a,b),count in pair_counts.items():
        normalised=count/max(1,min(freq[a],freq[b]))
        graph.add_edge(a,b,count=count,weight=min(1.,normalised))
    graph.graph['corpus_chunks']=corpus_count
    return graph

def save_graph(graph,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f: json.dump(nx.node_link_data(graph,edges='links'),f,ensure_ascii=False)

def read_graph(path):
    with open(path,encoding='utf-8') as f: return nx.node_link_graph(json.load(f),edges='links')

def graph_lexicon(graph):
    return {token for node in graph.nodes for token in node.split()}

def node_specificity(graph):
    return {v:math.log((1+graph.graph.get('corpus_chunks',1))/(1+graph.nodes[v]['frequency'])) for v in graph}

def _minmax(x):
    if not x:return {}
    lo=min(x.values());hi=max(x.values())
    return {key:(value-lo)/(hi-lo) if hi>lo else 0. for key,value in x.items()}

def structural_scores(graph):
    if not len(graph):return {}
    frequency=nx.get_node_attributes(graph,'frequency')
    top=max(1,math.ceil(len(graph)*.05))
    hubs=sorted(graph.nodes,key=lambda n:(-frequency.get(n,0),n))[:top]
    distance=nx.multi_source_dijkstra_path_length(graph,hubs,weight=None)
    cutoff=max(distance.values(),default=0)
    dist={v:distance.get(v,cutoff+1) for v in graph}
    deg=dict(graph.degree())
    specificity=node_specificity(graph)
    norm_dist=_minmax(dist);norm_deg=_minmax(deg);norm_spec=_minmax(specificity)
    return {v:(norm_dist[v]+norm_deg[v]+norm_spec[v])/3 for v in graph}

def link_entities(query, graph, nlp, encoder, threshold=.60, node_matrix=None, node_names=None):
    
    names=node_names or sorted(graph.nodes)
    if not names:return [],0.
    found=concepts(nlp(query))
    if not found:return [],0.
    if node_matrix is None:
        node_matrix=encoder.encode(names,normalize_embeddings=True,show_progress_bar=False)
    v=encoder.encode(found,normalize_embeddings=True,show_progress_bar=False)
    sims=np.asarray(v)@np.asarray(node_matrix).T
    matches=[]
    for i,name in enumerate(found):
        best=int(np.argmax(sims[i]));conf=float(sims[i,best])
        if conf>=threshold:matches.append({'query_entity':name,'node':names[best],'confidence':conf})
    return matches,(sum(m['confidence'] for m in matches)/len(matches) if matches else 0.)

def type_inputs(query, graph, nlp, encoder, nodes, vectors, scores, threshold=.60):
    from .features import lexical_score
    links,conf=link_entities(query,graph,nlp,encoder,threshold,vectors,nodes)
    kg=sum(scores[x['node']] for x in links)/len(links) if links else 0.
    return [conf*kg,lexical_score(query,graph_lexicon(graph))], {'links':links,'confidence':conf,'kg_score':kg}

def graph_evidence(query, links, graph, encoder, max_candidates=10, depth=2, edge_min=.2):
    """2-hop graph triples ranked by semantic relevance and normalized edge strength."""
    seeds=[m['node'] for m in links if m['node'] in graph]
    if not seeds:return []
    visit=set(seeds);front=set(seeds)
    for _ in range(depth):
        nxt={b for a in front for b in graph.neighbors(a) if graph[a][b].get('weight',0)>=edge_min}
        visit|=nxt;front=nxt
    triples=[]
    for a,b,attrs in graph.subgraph(visit).edges(data=True):
        if attrs.get('weight',0)<edge_min:continue
        text=f'{a} is related to {b}.'
        triples.append((f'kg:{min(a,b)}:{max(a,b)}',text,float(attrs['weight'])))
    if not triples:return []
    from .router import normalise_minmax
    qv=np.asarray(encoder.encode([query],normalize_embeddings=True))[0]
    tvec=np.asarray(encoder.encode([t[1] for t in triples],normalize_embeddings=True))
    sim=normalise_minmax((tvec@qv).tolist())
    weights=normalise_minmax([t[2] for t in triples])
    scored=[(t[0],t[1],.5*sim[i]+.5*weights[i]) for i,t in enumerate(triples)]
    return sorted(scored,key=lambda x:(-x[2],x[0]))[:max_candidates]
