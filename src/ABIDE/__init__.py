import os
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from abide_config import *


def load_data_fmri(site_id=None, harmonized=False):
    """
    Inputs
    site_id: str or None
        - if specified, splits will only contains index of subjects from the specified site
    harmonized: bool
        - whether or not to return combat harmonized X
    
    Returns
    X: np.ndarray with 823 subject samples, each sample consists of a 264 x 264 correlation matrix
    Y: np.ndarray with 823 subject samples, each sample has a one-hot encoded label (0: Normal, 1: Diseased)
    splits: np.ndarray with dimension 100 x 5 x 2
        - consists of indices for all subjects from all sites (targeted for supervised learning)
        - test indices of seed n = splits[n][0]
        - the train and val indices of seed n, fold k = splits[n][1][k][0] and splits[n][1][k][1]
    """
    if harmonized:
        X = np.load(HARMONIZED_X_PATH)
    else:
        X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    if site_id is None:
        splits = np.load(SPLITS_PATH, allow_pickle=True)
    else:
        splits = np.load(
            os.path.join(SSL_SPLITS_DIR, "{}.npy".format(site_id)), 
            allow_pickle=True
        )
    return X, Y, splits


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


def load_meta_df():
    return pd.read_csv(META_CSV_PATH)