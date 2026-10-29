import re
WORD=re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:\.\d+)?")
STOP={'a','an','the','is','are','of','on','for','in','at','to','and','or','from','with','as','by','what','why','how','does','do','did','it','was','were','which','that'}

def lexical_features(query, domain_vocab=()):
    tokens=WORD.findall(query)
    words=[w for w in tokens if re.search('[A-Za-z]',w)]
    content=[w for w in words if w.lower() not in STOP]
    avg_len=(sum(len(w) for w in content)/len(content)) if content else 0
    terms=set(v.lower() for v in domain_vocab)
    domain_hits=sum(w.lower() in terms for w in words)
    technical=sum(bool(re.search(r'\d|[A-Z]{2,}|-|[()\[\]{}=+*/%]',w)) for w in tokens)
    # Marker punctuation not captured by WORD; add symbolic markers by text scan.
    technical+=len(re.findall(r'[()\[\]{}=+*/%]',query))
    n=max(1,len(tokens))
    return [min(1.0, avg_len/12), sum(len(w)>7 for w in words)/max(1,len(words)),
            domain_hits/max(1,len(words)), min(1.0,len(tokens)/25), min(1.0,technical/n)]

def lexical_score(query, domain_vocab=()):
    return sum(lexical_features(query,domain_vocab))/5
