import collections, numpy

def sort(subject_ids, ages, *arrays):

    '''

    Description:
        We wish to model outcome y_i in Y given features x_p in X,
        where p < i. The problem is X and Y need to be first sorted by
        subject ids, then sorted age.

    Input:
        subject_ids - numpy array of shape (N), the subject ids
            corresponding to the features.
        ages - numpy array of shape (N), the ages at which the
            features are recorded.
        *arrays - numpy arrays of shape (N, *), to be grouped and
            sorted by corresponding subject_ids and ages.
    
    Output:
        chronosorted_arrays - numpy array of (N', S, *) shape. Sort X
            by subject id, then age. S is the longest sequence encountered.
            N' <= N.
        mask - numpy array of (N, S) shape. 0 signals padding, 1 signals 
            actual data.
    
    '''
    arrays = _expand_if_necessary(arrays)
    subject_bins = _separate_by_subject(
        [numpy.expand_dims(subject_ids, axis=1)] + list(arrays),
        subject_ids,
        ages
    )
    age_sorted = _sort_age_per_bin(subject_bins)
    return _fix_arrays(age_sorted)

def _expand_if_necessary(arrays):
    return [
        numpy.expand_dims(a, axis=-1)
        if len(a.shape)==1 else a
        for a in arrays
    ]

def _separate_by_subject(arrays, subject_ids, ages):
    bins = collections.defaultdict(list)
    for i, (sid, age) in enumerate(zip(subject_ids, ages)):
        bins[sid].append((age, [arr[i] for arr in arrays]))
    return list(bins.values())

def _sort_age_per_bin(subject_bins):
    return [
        [
            row[1]
            for row in sorted(
                group,
                key=lambda a: a[0]
            )
        ] for group in subject_bins
    ]

def _fix_arrays(age_sorted):
    assert age_sorted
    assert age_sorted[0]

    N = len(age_sorted)
    S = max(map(len, age_sorted))

    # Each output array will have shape (N, S, *)
    array_count = len(age_sorted[0][0])

    out_arrays = []
    mask = numpy.zeros((N, S), dtype=numpy.bool)
    for arr_i in range(array_count):
        D = len(age_sorted[0][0][arr_i])
        out = numpy.zeros((N, S, D))
        for subject_id, l in enumerate(age_sorted):
            for age_i, row in enumerate(l):
                out[subject_id, age_i] = row[arr_i]

                if arr_i:
                    assert mask[subject_id, age_i]
                else:
                    mask[subject_id, age_i] = 1

        if D == 1:
            out = numpy.squeeze(out, axis=-1)
        out_arrays.append(out)
    return out_arrays, mask

if __name__ == "__main__":
    
    X = numpy.array([
        [338, 19, 52],
        [703, 19, 26],
        [900, 19, 4],
        [385, 19, 20],
        [986, 19, 64],
        [384, 19, 78],
        [455, 19, 89],
        [927, 3, 83],
        [813, 3, 53],
        [281, 3, 95],
        [559, 3, 67],
        [779, 3, 87],
        [868, 3, 98],
        [460, 3, 54],
        [390, 3, 30],
        [592, 3, 92],
        [255, 0, 97],
        [982, 0, 66],
        [248, 0, 9],
        [399, 0, 83],
        [280, 8, 82],
        [535, 8, 21],
        [407, 8, 7],
        [590, 8, 62],
        [777, 8, 46],
        [401, 8, 47],
        [707, 8, 56],
        [583, 13, 1],
        [977, 13, 57],
        [896, 13, 24],
        [978, 13, 29]
    ])
    
    numpy.random.shuffle(X)
    
    func = lambda x: x//100
    Y = func(X[:,0])
    
    (Xp, Yp), mask = sort(X[:,1], X[:,2], X, Y)
    
    masked_X = Xp * numpy.expand_dims(mask, -1)
    masked_Y = Yp * mask
    assert (func(masked_X[:,:,0]) == masked_Y).all()
    
    for row in range(len(Xp)):
        arb_row = Xp[row,:,2][mask[row,:]]
        assert (arb_row == numpy.sort(arb_row)).all()
