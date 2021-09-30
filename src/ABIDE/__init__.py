import os
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from abide_config import *


def load_data_fmri(harmonized=False):
    """
    Inputs
    site_id: str or None
        - if specified, splits will only contains index of subjects from the specified site
    harmonized: bool
        - whether or not to return combat harmonized X
    
    Returns
    X: np.ndarray with 823 subject samples, each sample consists of a 264 x 264 correlation matrix
    Y: np.ndarray with 823 subject samples, each sample has a one-hot encoded label (0: Normal, 1: Diseased)
    """
    if harmonized:
        X = np.load(HARMONIZED_X_PATH)
    else:
        X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    return X, Y


def get_splits(site_id=None, test=False):
    if site_id is None:
        path = SPLIT_TEST_PATH if test else SPLIT_CV_PATH
    else:
        path = "{}_test.npy".format(site_id) if test else "{}_cv.npy".format(site_id)
        path = os.path.join(SSL_SPLITS_DIR, path)
    splits = np.load(path, allow_pickle=True)
    return splits


def get_ages_and_genders():
    """
    ages: np.array of float representing the age of subject when the scan is obtained
    gender: np.array of int representing the subject's gender
        - 0: Male
        - 1: Female
    """
    meta_df = pd.read_csv(META_CSV_PATH)
    ages = np.array(meta_df["AGE_AT_SCAN"])
    genders = np.array(meta_df["SEX"] - 1)
    return ages, genders


def get_sites():
    meta_df = pd.read_csv(META_CSV_PATH)
    sites = np.array(meta_df["SITE_ID"])
    return sites


def load_meta_df():
    return pd.read_csv(META_CSV_PATH)


def get_labelling_standards():
    """
    "gold standard" diagnostic instruments
    - Autism Diagnostic Observation Schedule (ADOS)
    - Autism Diagnostic Interview - Revised (ADI-R)

    0: clinical judgement + "gold standard" diagnostic instruments
    1: clinical judgement only
    2: "gold standard" diagnostic instruments only
    """
    return {
        "CALTECH": 0,
        "CMU": 0,
        "KKI": 0,
        "LEUVEN_1": 1, 
        "LEUVEN_2": 1, 
        "MAX_MUN": 0,
        "NYU": 0,
        "OHSU": 0,
        "OLIN": 2, 
        "PITT": 0,
        "SBL": 1,
        "SDSU": 0,
        "STANFORD": 2, 
        "TRINITY": 0,
        "UCLA_1": 2, 
        "UCLA_2": 2,
        "UM_1": 0,
        "UM_2": 0,
        "USM": 0,
        "YALE": 0
    }
