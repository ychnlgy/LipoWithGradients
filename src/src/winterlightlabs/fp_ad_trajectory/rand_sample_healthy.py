import src

import collections, torch

def sample(U, X, Y, M, T, P, observation_dropout=0.0, feature_dropout=0.0, std_max=0.0, iterations=0, disable=False):
    
    '''
    
    Description:
        We wish to sample only a random subset of the observations made
        in the healthy moments of the subjects' lives. We also wish
        to randomly sample new subsets per epoch, applying some level
        of Gaussian noise and dropout to the features each time.
        
        Therefore, note there are two occasions of dropout: the first is
        when we wish to select a random subset from all healthy observations,
        and the second to when we wish to zero out features randomly.
        
        This function will also remove any subjects which do not have any
        observations made when they were healthy.
    
    Input:
        U - torch Tensor of shape (N, S), unique id.
        X - torch Tensor of shape (N, S, D), where N is the batch size,
            S is the padded sequence length and D is the number of features.
        Y - torch Tensor of shape (N, S), diagnosis of AD, with value 0 
            corresponding to observations made before formal diagnosis and
            value 1 corresponding to observations made at or after formal
            AD diagnosis.
        M - torch Tensor of shape (N, S), with values 0 corresponding to 
            padded regions and 1 otherwise.
        T - torch Tensor of shape (N, S), ages of the subjects.
        P - torch Tensor of shape (N, K), where K is the number of parameters
            of the sigmoid. K = 4 usually.
        observation_dropout - float dropout for the healthy observations.
        feature_dropout - float dropout for the features.
        std_max - float number of standard deviations to apply noise.
        iterations - int number of random permutations.
    
    Output:
        U' - torch Tensor of shape (N'*iterations, S).
        X' - torch Tensor of shape (N'*iterations, S, D), subset of healthy observations.
            N' <= N because certain subjects with no healthy observations are
            removed.
        Y' - torch Tensor of shape (N'*iterations, S).
        M' - torch Tensor of shape (N'*iterations, S).
        T' - torch Tensor of shape (N'*iterations, S).
        P' - torch Tensor of shape (N'*iterations, K).
        I' - torch Tensor of shape (N'), indices to the original dataset (X, Y, M, T, D).
            This is used for plotting the corresponding original values.
    
    '''
    out = collections.defaultdict(list)

    if not disable:
        M = (M == 1) & (Y == 0) # these are the healthy observations.
    
    for i in range(iterations):

        M_dropped = src.dataset.corrupt.apply_dropout(M, observation_dropout)
        Up, Xp, Yp, Mp, Tp, Pp, Ip = _collect_and_squeeze(U, X, Y, M_dropped, T, P)
        
        # corrupt the features using noise and dropout
        Xp = src.dataset.corrupt.apply_gaussian_noise(std_max, Xp, X_all=X)
        Xp = src.dataset.corrupt.apply_dropout(Xp, feature_dropout)

        #Pp = src.dataset.corrupt.apply_gaussian_noise(std_max, Pp, X_all=P)
        
        _zip_append([Up, Xp, Yp, Mp, Tp, Pp, Ip], out)
        
    return _merge(out)

def _zip_append(tensors, defaultdictlist):
    for i, t in enumerate(tensors):
        defaultdictlist[i].append(t)

def _merge(defaultdictlist):
    N = len(defaultdictlist)
    out = [None] * N
    for i in range(N):
        out[i] = torch.cat(defaultdictlist[i])
    return out

def _collect_and_squeeze(U, X, Y, M, T, P):
    
    '''
    
    Description:
        Mask M is fragmented. We must select the values or vectors corresponding
        to the mask, then place them side-by-side in sequence.
    
    '''
    
    out = []
    keep = M.sum(dim=1) > 0
    all_t = [U, X, Y, M, T]
    for i, t in enumerate(all_t): 
        buffer = torch.zeros_like(t)
        for i, (row, mask_row) in enumerate(zip(t, M)):
            
            # The following line is where we place selected values
            # or vectors side-by-side.
            buffer[i,:mask_row.sum()] = row[mask_row==1]
            
        out.append(buffer[keep])

    out.append(P[keep])

    # These are indices pointing to the original data location.
    I = torch.arange(len(M)).long()[keep]
    out.append(I)
    
    return out
