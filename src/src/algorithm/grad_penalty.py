import torch

import src

MAX_TRIES = 100
EPS = 1e-8

def lipschitz_max_grad(net, X_observe, X_target, Y_target, k=1):
    '''

    Description:
        Penalizes the model if it does not have a K-Lipschitz
        maximum gradient-norm  for any interpolant between samples
        from the target and observed distributions.

        Note for multidimensional inputs, K-Lipschitz quality is:

            ||f(x1) - f(x2)|| <= K*||x1 - x2||

    Input:
        net - torch.nn.Module, the training model.
        X_observe - torch Tensor of size (N, D), samples from the
            observed distribution.
        X_target - torch Tensor of size (M, D), samples from the
            target distribution.
        Y_target - torch Tensor of size (M, T), output for X_target.
        k - float, target Lipschitz constant.

    Output:
        loss - torch Tensor of shape (0), loss value from which
            backpropogation should begin.

    '''
    X_choice, Y_choice = _rand_diff_choice(X_observe, X_target, Y_target)
    X_interp = _rand_blend(X_observe, X_choice)
    Y_interp = net(X_interp)
    grad = torch.autograd.grad(
        [Y_interp.mean()],
        net.parameters(),
        create_graph=True,
        only_inputs=True
    )[0]
    return ((k-grad.norm(p=2))**2).mean()

# === PRIVATE ===

def _rand_blend(X1, X2):
    assert X1.size() == X2.size()
    alpha = torch.rand_like(X1) * (1-EPS)
    return alpha*X1 + (1-alpha)*X2

def _rand_diff_choice(X_observe, X_target, Y_target):
    N = X_target.size(0)
    M = X_observe.size(0)
    assert N >= M
    
    I = src.tensortools.rand_indices(N)[:M]
    X_choice = X_target[I].clone()
    Y_choice = Y_target[I].clone()
    too_close = _check_too_close(X_observe, X_choice)
    
    for i in range(MAX_TRIES):
        m = too_close.long().sum().item()
        if m == 0:
            break
        
        I = src.tensortools.rand_indices(N)[:m]
        X_choice[too_close] = X_target[I].clone()
        Y_choice[too_close] = Y_target[I].clone()
        too_close = _check_too_close(X_observe, X_choice)
        
    assert i < MAX_TRIES # ERROR: No close matches were found!
    return X_choice, Y_choice

def _check_too_close(X1, X2):
    return (X1 - X2).norm(p=2, dim=1) < EPS
