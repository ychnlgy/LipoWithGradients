import torch, numpy

from .. import data
from . import chronosort, extra, sigmoidfit_t0, rand_sample_healthy

BOUNDS = [
    # c, k,  x0, y0
    ( 0, 0,   0,  0),
    ( 1, 2, 100,  0.01)
]

TOL = 1e-6

def load(
    datadir,
    prediagnosis_years,
    disable_healthy_only=False,
    boruta=False,
    observation_dropout=0.0,
    feature_dropout=0.0,
    std_max=0.0,
    iterations=0,
    fp_data = None
):

    '''

    Combines all code in this directory into a single function.

    Consult the files for their documentation and outputs.

    '''

    if fp_data is None:
        _, (X, pure_Y, subject_ids, ages), _ = data.load(datadir, boruta)
    else:
        (X, pure_Y, subject_ids, ages) = fp_data

    ad_trajectories, age_of_diag = extra.get_AD_diagnosis_trajectory(
        datadir,
        subject_ids,
        ages,
        prediagnosis_years
    )

    (U, X, T, Y, A), M = chronosort.sort(
        subject_ids,
        ages,
        X,
        ages,
        ad_trajectories,
        age_of_diag
    )

    A = A[:,0]

    Ur, Xr, Yr, Mr, Tr, Ar = tuple(
        map(
            torch.from_numpy,
            [
                U.astype(numpy.int64),
                X.astype(numpy.float32),
                Y.astype(numpy.int64),
                M.astype(numpy.uint8),
                T.astype(numpy.float32),
                A.astype(numpy.float32)
            ]
        )
    )

    healthy_samples = rand_sample_healthy.sample(
        Ur, Xr, Yr, Mr, Tr, Ar,
        observation_dropout,
        feature_dropout,
        std_max,
        iterations,
        disable = disable_healthy_only
    )

    return (
        (Ur, Xr, Yr, Mr, Tr, Ar), # reference original all data.
        healthy_samples # (U, X, Y, M, T, A, I) << don't forget extra array!
    )
