import torch, numpy

def convert_size(data, size):
    N, C, W, H = data.size()
    X, Y = size
    CX = (X - W)//2
    CY = (Y - H)//2
    out = torch.zeros(N, C, *size)
    out[:,:,CX:CX+W,CY:CY+H] = data
    return out

def create_loader(batch, X, Y=None):
    if Y is None:
        Y = torch.zeros(len(X)).long()
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    return dataloader

def pillow_to_tensor(dataset, resize=None):

    if resize is None:
        resize_fn = lambda x: x
    else:
        resize_fn = lambda x: x.resize(resize)

    data, labs = [], []
    for d, l in dataset:
        d = resize_fn(d)
        data.append(numpy.array(d))
        labs.append(l)
    
    data = torch.from_numpy(numpy.array(data))
    labs = torch.from_numpy(numpy.array(labs))
    
    assert 255.0 >= data.max() > 200.0
    assert 50.0 > data.min() >= 0.0
    
    data = data.permute(0, 3, 1, 2).float()/255.0
    
    return data, labs.long()
