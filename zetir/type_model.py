"""Pairwise ranker, two inputs, 16 hidden, sigmoid, margin loss."""
import numpy as np

def train_ranker(train_pairs, val_pairs, output_path, seed=42, epochs=50, patience=5, margin=.1):
    import torch
    from torch import nn
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True, warn_only=True)
    class Network(nn.Module):
        def __init__(self):
            super().__init__(); self.net=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Dropout(.1),nn.Linear(16,1),nn.Sigmoid())
        def forward(self,x):return self.net(x).flatten()
    def inputs(p,kind):
        x=p[kind]
        if len(x)!=2:raise ValueError('Every query must have [gated_kg, lexical] feature')
        return x
    def arr(pairs,kind):return torch.tensor([inputs(p,kind) for p in pairs],dtype=torch.float32)
    ntr,etr=arr(train_pairs,'novice'),arr(train_pairs,'expert')
    nv,ev=arr(val_pairs,'novice'),arr(val_pairs,'expert')
    model=Network(); opt=torch.optim.AdamW(model.parameters(),lr=.001)
    rng=np.random.default_rng(seed);best_loss=float('inf'); stale=0; best=None
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(ntr)),max(1,int(np.ceil(len(ntr)/32)))):
            if not len(ids): continue
            loss=torch.relu(margin-model(etr[ids])+model(ntr[ids])).mean()
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): v=float(torch.relu(margin-model(ev)+model(nv)).mean())
        if v<best_loss-1e-7:
            best_loss=v; stale=0;best={k:t.detach().clone() for k,t in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience:break
    model.load_state_dict(best)
    from pathlib import Path
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'seed':seed,'validation_pairwise_loss':best_loss},p)
    return {'best_validation_loss':best_loss,'epochs_run':epoch+1}

def load_ranker(path):
    import torch
    from torch import nn
    model=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Dropout(.1),nn.Linear(16,1),nn.Sigmoid())
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    state={k.removeprefix('net.'):v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state);model.eval()
    return lambda features:float(model(torch.tensor([features],dtype=torch.float32)).item())
