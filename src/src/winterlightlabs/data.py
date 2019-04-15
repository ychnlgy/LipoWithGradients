# TODO: Check below for code to clean up!!!

DATADIR = None

def load(datadir, boruta=False):
    global DATADIR
    DATADIR = datadir
    _TEMP_CONTAINER.clear()
    boruta_features = _get_boruta(datadir, boruta)
    sys.stderr.write("Loading:\n")
    return (
        _get_data(datadir, DEMENTIABANK_INFO, boruta_features, skiphash=True),
        _get_data(datadir, FAMOUSPEOPLE_INFO, boruta_features, skiphash=True),
        _get_data(datadir, HEALTHYAGING_INFO, boruta_features)
    )

# === PRIVATE ===

import pandas, numpy, os, json, sys
from .fp_ad_trajectory import extra
from .constants import *

FILE_TEMPLATE = "{name}_v{version}_normalized.csv"

HEADER_SUBJECTID = "original_subject_id"
HEADER_AGE = "age"
HEADER_DIAG = "diagnoses"

INDEX_FEATURESTART = 25
FEATURES = 482
INDEX_FEATURES = slice(INDEX_FEATURESTART, INDEX_FEATURESTART+FEATURES)

class DatasetInfo:

    ORDER = 0
    _HASH = 100000 # not expecting more than 100000 datasets

    def __init__(self, name, version, labelmap, force_change=None):
        self.name = name
        self.version = version
        self.labelmap = labelmap
        self.force_change = force_change
        
        self.order = DatasetInfo.ORDER
        DatasetInfo.ORDER += 1

DEMENTIABANK_INFO = DatasetInfo(
    name = "dementiabank",
    version = 33,
    labelmap = {
        "ad":AD_VALUE,
        "hc": HC_VALUE,
        "mci":MCI_VALUE,
        "unknown":numpy.nan,
        "dementia_vascular":numpy.nan
    }
)

TRUMP = 528
HC = "hc"

def _famouspeople_preprocess(U, D):
    _trump_is_hc(U, D)
    _fix_fp_nans(U, D)

def _fix_fp_nans(U, D):
    # Fix all NaNs with Jordan's table
    fp_diag = extra.get_subject_id_to_diagnosis_map(DATADIR)
    for i, d in fp_diag.items():
        D[U == int(i)] = d

def _trump_is_hc(U, D):
    # Trump is not formally diagnosed with AD
    D[U == TRUMP] = HC

FAMOUSPEOPLE_INFO = DatasetInfo(
    name = "famouspeople",
    version = 34,
    labelmap = {"ad":AD_VALUE, "hc":HC_VALUE, "nan":numpy.nan},
    force_change = _famouspeople_preprocess
)

HEALTHYAGING_INFO = DatasetInfo(
    name = "healthyaging",
    version = 34,
    labelmap = {"nan":HC_VALUE}
)

BORUTA_FILE = "boruta_features.json"

def _get_boruta(datadir, boruta):
    "Returns None if boruta is False."
    if boruta:
        borutaf = os.path.join(datadir, BORUTA_FILE)
        assert os.path.isfile(borutaf)
        with open(borutaf) as f:
            out = numpy.array(json.load(f))

        global FEATURES
        FEATURES = len(out)
        return out

def _get_data(datadir, info, boruta_features, skiphash=False):
    '''

    Output:
        X - numpy array of shape (N, D), the features.
        Y - numpy array of shape (N), int AD diagnoses.
        U - numpy array of shape (N), int unique ids.
        T - numpy array of shape (N), float ages.

    '''
    
    path = _get_filepath(datadir, info)
    pand = pandas.read_csv(path)

    sys.stderr.write(" << %s\n" % path)
    
    U = pand[HEADER_SUBJECTID].values
    T = pand[HEADER_AGE].values
    D = pand[HEADER_DIAG].values

    H = boruta_features if boruta_features is not None else pand.columns[INDEX_FEATURES]
    X = pand[H].values

    X = _fix_nan_features(X)

    if info.force_change is not None:
        info.force_change(U, D)

    U = _make_unique(U, info.order, skiphash)    
    Y = _process_str_diagnoses(D, info.labelmap, T)

    X, Y, U, T = _filter_nan_Y(X, Y, U, T)
    
    return (
        X.astype(numpy.float32),
        Y.astype(numpy.int64),
        U.astype(numpy.int64),
        T.astype(numpy.float32)
    )

def _fix_nan_features(X):
    assert len(X.shape) == 2
    keep = [_calc_mean_feature(X, i) for i in range(X.shape[1])]
    return numpy.stack(keep, axis=1)

def _calc_mean_feature(X, i):
    column = X[:,i].copy()
    finite = numpy.isfinite(column)
    check = column[finite]
    if not len(check):
        return numpy.zeros(len(X))
    else:
        column[numpy.logical_not(finite)] = check.mean()
        assert numpy.isfinite(column).all()
        return column

def _filter_nan_Y(X, Y, U, T):
    i = numpy.isfinite(Y)
    return X[i], Y[i], U[i], T[i]

def _process_str_diagnoses(D, labelmap):
    return numpy.array([labelmap[str(v)] for v in D])

def _process_str_diagnoses(D, labelmap, T):
    return numpy.array([
        _convert_str_to_int(labelmap, d, t) for d, t in zip(D, T)
    ])

def _convert_str_to_int(labelmap, d, t):
    # TODO: Check this area for code to clean up!!!
    try:
        d = json.loads(d)
        assert len(d) == 1
        d = d[0]
        label = d["name"]
        diag_year = d["diagnosis_year"]
        if diag_year is not None:
            input(diag_year)
    except:
        label = str(d)
    assert label in labelmap
    return labelmap[label]

def _make_unique(U, order, skiphash):
    "Transforms ids in U into globally unique ids, given the dataset order."
    if not skiphash:
        U = U*DatasetInfo._HASH + order
    _check_truly_unique(U)
    return U

_TEMP_CONTAINER = set()

def _check_truly_unique(U):
    U = set(U)
    b4 = len(_TEMP_CONTAINER)
    _TEMP_CONTAINER.update(U)
    assert len(_TEMP_CONTAINER)-len(U) == b4

def _get_filepath(datadir, info):
    fname = FILE_TEMPLATE.format(name=info.name, version=info.version)
    fpath = os.path.join(datadir, fname)
    assert os.path.isfile(fpath)
    return fpath

    

