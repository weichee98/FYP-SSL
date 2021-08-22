import os
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from abide_config import *


def load_data_fmri(data_dir=MAIN_DIR):
    """
    X: np.ndarray with 823 subject samples, each sample consists of a 264 x 264 correlation matrix
    Y: np.ndarray with 823 subject samples, each sample has a one-hot encoded label (0: Normal, 1: Diseased)
    splits: np.ndarray with dimension 100 x 5 x 2
        - test indices of seed n = splits[n][0]
        - the train and val indices of seed n, fold k = splits[n][1][k][0] and splits[n][1][k][1]
    """
    X = np.load(os.path.join(data_dir, X_FNAME)) # (823, 264, 264)
    Y = np.load(os.path.join(data_dir, Y_FNAME)) # (823, 2)
    splits = np.load(os.path.join(data_dir, SPLITS_FNAME), allow_pickle=True)
    ssl_splits = np.load(os.path.join(data_dir, SSL_SPLITS_FNAME), allow_pickle=True)
    return X, Y, splits, ssl_splits


def get_ages_and_genders(meta_csv_path=META_CSV_PATH):
    """
    ages: np.array of float representing the age of subject when the scan is obtained
    gender: np.array of int representing the subject's gender
        - 0: Male
        - 1: Female
    """
    meta_df = pd.read_csv(meta_csv_path)
    ages = np.array(meta_df["AGE_AT_SCAN"])
    genders = np.array(meta_df["SEX"] - 1)
    return ages, genders


def load_meta_df(meta_csv_path=META_CSV_PATH):
    return pd.read_csv(meta_csv_path)