import os, pandas, numpy, csv

from ..constants import AD_VALUE, PRE_DIAGNOSIS_VALUE, HC_VALUE

INFO_FILE = "famouspeople_info.csv"
INFO_HEADERS = [
    "Person:Name",
    "Person:Date of Birth",
    "Diagnosis:Date",
    "Control:Name"
]

INFO_CONTROL_DIAG = 150

SUBJECT_FILE = "famouspeople_subjects.csv"
SUBJECT_ID = "id"
SUBJECT_FIRSTNAME = "firstname"
SUBJECT_LASTNAME = "lastname"
SUBJECT_DIAG = "name"

NAME_TYPOS = {
    "Ronald Regan": "Ronald Reagan",
    "George Bush Sr.": "George Bush Sr",
    "Ruth  Bader Ginsburg": "Ruth Bader Ginsberg"
}

def get_AD_diagnosis_trajectory(datadir, subject_ids, ages, prediagnosis_years):

    '''

    Input:
        datadir - str folder containing the data.
        subject_ids - numpy array of shape (N), unique ids.
        ages - numpy array of shape (N), ages.
        prediagnosis_years - int number of years prior to diagnosis

    Output:
        Y - numpy array of shape (N), with value 0 for
            ages < age of AD diagnosis - prediagnosis, 2 for
            age of diagnosis - prediagnosis < ages < age of diagnosis,
            and 1 for all ages post-diagnosis.

    '''
    diag_age_map = _create_diag_age_map(datadir)
    age_of_diag = numpy.array([diag_age_map[i] for i in subject_ids])
    Y = (age_of_diag <= ages).astype(numpy.int64)
    Y[Y==0] = PRE_DIAGNOSIS_VALUE
    Y[Y==1] = AD_VALUE
    Y[ages < (age_of_diag - prediagnosis_years)] = HC_VALUE
    return Y.astype(numpy.float32), age_of_diag

def get_subject_id_to_diagnosis_map(datadir):
    return {i:d for _, i, d in _parse_subjects(datadir)}

def _create_diag_age_map(datadir):
    "Returns dict of {original_subject_id: age_of_AD_diagnosis}."
    info = _parse_info(datadir)
    out = {}
    for name, original_subject_id, _ in _parse_subjects(datadir):
        out[int(original_subject_id)] = info[name]
    return out

def _parse_subjects(datadir):
    data = _parse_csv(os.path.join(datadir, SUBJECT_FILE))
    for i, fname, lname, diag in zip(
        data[SUBJECT_ID],
        data[SUBJECT_FIRSTNAME],
        data[SUBJECT_LASTNAME],
        data[SUBJECT_DIAG]
    ):
       
        name = " ".join([fname, lname])
        if name in NAME_TYPOS: # there is a typo in this dataset
            name = NAME_TYPOS[name]
        yield name, i, diag

def _parse_info(datadir):
    data = _parse_csv(os.path.join(datadir, INFO_FILE), header_lines=2)
    out = {}
    for (
        subject_name,
        subject_dob,
        subject_diag,
        control_name
    ) in zip(*[data[h] for h in INFO_HEADERS]):
        if subject_name:
            try:
                dob = _extract_year_from_date(subject_dob)
                diag = _extract_year_from_date(subject_diag)
                out[subject_name] = diag - dob
            except ValueError:
                out[subject_name] = INFO_CONTROL_DIAG # this is for the Trump example.
            out[control_name] = INFO_CONTROL_DIAG
    return out

def _parse_csv(fname, header_lines=1):
    with open(fname) as f:
        csvf = csv.reader(f)
        out, headers = _merge_header_lines(csvf, header_lines)
        for line in csvf:
            for h, v in zip(headers, line):
                out[h].append(v)
        return out

def _merge_header_lines(csvf, header_lines):
    if header_lines == 1:
        headers = next(csvf)
    else:
        primary = next(csvf)
        secondary = next(csvf)
        bases = []
        main_header = None
        for h in primary:
            if h:
                main_header = h
            bases.append(main_header)
        headers = [("%s:%s" % h) for h in zip(bases, secondary)]
    out = {h:[] for h in headers}
    return out, headers

def _extract_year_from_date(date):
    year, month, day = date.split("-")
    return int(year)
