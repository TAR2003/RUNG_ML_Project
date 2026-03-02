import torch
import tqdm
import torch.nn.functional as F

from utils import accuracy, device


'''
def fit(model: torch.nn.Module, A, X, y, train_idx, val_idx, **kwargs):
    
    #Train model on graph A, X, using train_idx and val_idx
    #given the settings in kwargs
    
    optimizer = torch.optim.Adam(model.parameters(), **{key: kwargs[key] for key in kwargs if key in ['lr', 'weight_decay']})

    for i in tqdm.trange(kwargs['max_epoch'] if 'max_epoch' in kwargs else 3000):
        loss = F.cross_entropy(model(A, X)[train_idx], y[train_idx])
        loss.backward(retain_graph=False, create_graph=False)
        optimizer.step()

        if i % 10 == 0:
            print(accuracy(model(A, X)[val_idx], y[val_idx]))
'''

def fit(model: torch.nn.Module, A, X, y, train_idx, val_idx, **kwargs):
    
    #Train model on graph A, X, using train_idx and val_idx
    #given the settings in kwargs
    
    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device) if isinstance(train_idx, torch.Tensor) else train_idx
    val_idx = val_idx.to(device) if isinstance(val_idx, torch.Tensor) else val_idx

    optimizer = torch.optim.Adam(model.parameters(), **{key: kwargs[key] for key in kwargs if key in ['lr', 'weight_decay']})


    for i in tqdm.trange(kwargs['max_epoch'] if 'max_epoch' in kwargs else 3000):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(A, X)[train_idx], y[train_idx])
        #loss = F.nll_loss(model(A, X)[train_idx], y[train_idx])
        #loss.backward(retain_graph=False, create_graph=False)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            model.eval()
            print(accuracy(model(A, X)[val_idx], y[val_idx]))
        

