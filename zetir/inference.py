from .router import Router, fuse_candidates
from .graph import type_inputs,graph_evidence

PROMPTS={
'dense':'Give a short, clear explanation in everyday language. Explain technical terms when needed. Use only the provided evidence.',
'hybrid':'Start with a simple explanation and then give the necessary technical detail. Use only the provided evidence.',
'graph':'Give a detailed answer using appropriate domain terms, mechanisms, and relations. Use only the provided evidence.'}

def route_and_retrieve(query, graph, dense_index, nlp, encoder, ranker, node_names, node_vectors, structural, lower=.38,upper=.45):
    vals,linking=type_inputs(query,graph,nlp,encoder,node_names,node_vectors,structural)
    tau=ranker(vals);router=Router(lower,upper);route=router.route(tau)
    ds=dense_index.search(query,5) if route in ('dense','hybrid') else []
    gs=graph_evidence(query,linking['links'],graph,encoder) if route in ('graph','hybrid') else []
    # If pure graph returns no evidence, fallback avoids empty context; log as fallback.
    fallback=route=='graph' and not gs
    if fallback:ds=dense_index.search(query,5)
    items=fuse_candidates(ds,gs,router.graph_weight(tau),top_k=5) if route=='hybrid' else [
        {'id':r[0],'text':r[1],'score':r[2]} for r in (ds if route=='dense' or fallback else gs)[:5]]
    return {'tau':tau,'route':route,'fallback':fallback,'feature_values':vals,
            'linking':linking,'evidence':items}

def load_generator(model_name='microsoft/Phi-3-mini-4k-instruct',max_input_tokens=2048):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError('4-bit Phi-3 generation requires CUDA; CPU is not supported by this preset')
    tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=False)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quant,device_map='auto',trust_remote_code=False)
    return model,tok

def generate_answer(query, context, route, model, tokenizer, input_limit=2048,output_limit=256):
    ev='\n'.join(f'[{i+1}] {r["text"]}' for i,r in enumerate(context))
    prompt=f'{PROMPTS[route]}\n\nEvidence:\n{ev}\n\nQuestion: {query}\nAnswer:'
    inputs=tokenizer(prompt,return_tensors='pt',truncation=True,max_length=input_limit).to(model.device)
    out=model.generate(**inputs,max_new_tokens=output_limit,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
