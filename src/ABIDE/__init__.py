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


def labelling_standard(site_id):
    """
    "gold standard" diagnostic instruments
    - Autism Diagnostic Observation Schedule (ADOS)
    - Autism Diagnostic Interview - Revised (ADI-R)

    1: clinical judgement + "gold standard" diagnostic instruments
    2: clinical judgement only
    3: "gold standard" diagnostic instruments only
    """
    if site_id in ["LEUVEN_1", "LEUVEN_2", "SBL"]:
        return 2
    if site_id in ["OLIN", "STANFORD", "UCLA_1", "UCLA_2"]:
        return 3
    return 1


def get_label_standard_mask(sites, target_label_standard=None):
    """
    sites: np.array of site_id
    target_label_standard: 1, 2, 3 or None
    """
    if target_label_standard is None:
        return np.ones_like(sites, dtype=bool)
    elif not isinstance(target_label_standard, int):
        raise TypeError(
            "target_label_standard can only be integer or None"
        )
    elif target_label_standard < 1 or target_label_standard > 3:
        raise ValueError(
            "target_label_standard must be within [1, 3], but given {}"
            .format(target_label_standard)
        )
    standards = np.vectorize(labelling_standard)(sites)
    return standards == target_label_standard
    
