import torch

def leave_one_out(*tensors):
    assert tensors
    lengths = set(map(len, tensors))
    assert len(lengths) == 1
    N = lengths.pop()
    for i in range(N):
        test_tensors = [t[i:i+1] for t in tensors]
        train_tensors = [_get_train(v, i) for v in tensors]
        yield (train_tensors, test_tensors)

def _get_train(v, i):
    return torch.cat([v[:i], v[i+1:]])
