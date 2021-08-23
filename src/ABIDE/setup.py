import os
import time
import warnings
import numpy as np
import pandas as pd
from abide_config import *
from contextlib import contextmanager
from neuroCombat import neuroCombat
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@contextmanager
def log_time(description):
    print("[{}] started".format(description))
    start = time.time()
    yield
    end = time.time()
    print("[{}] completed in {:.3f} s".format(description, end - start))


def get_processed_corr_mat_file_ids(corr_mat_dir):
    file_ids = []
    for _, _, files in os.walk(corr_mat_dir):
        for filename in files:
            if filename.endswith(".npy"):
                file_ids.append(filename[:-10])
    return file_ids


def get_file_path(corr_mat_dir, dx_group, file_id):
    filename = "{}_power.npy".format(file_id)
    if dx_group == 1:
        return os.path.join(corr_mat_dir, "diseased", filename)
    else:
        return os.path.join(corr_mat_dir, "normal", filename)


def extract_data(main_dir, corr_mat_dir, meta_csv_path):    
    meta_df = pd.read_csv(meta_csv_path, index_col=0)
    meta_df = meta_df.drop(["Unnamed: 0.1", "subject"], axis=1)
    file_ids = get_processed_corr_mat_file_ids(corr_mat_dir)

    meta_df["PROCESSED"] = meta_df["FILE_ID"].apply(lambda x: x in file_ids)
    processed_df = meta_df[meta_df["PROCESSED"]].sort_values("SUB_ID")
    processed_df = processed_df.drop("PROCESSED", axis=1)
    processed_df["FILE_PATH"] = processed_df[["DX_GROUP", "FILE_ID"]].apply(
        lambda x: get_file_path(corr_mat_dir, x["DX_GROUP"], x["FILE_ID"]), axis=1
    )
    processed_df.to_csv(META_CSV_PATH, header=True, index=False)

    X = np.array([np.load(fname) for fname in processed_df["FILE_PATH"]])
    X = np.nan_to_num(X)
    Y = np.array(2 - processed_df["DX_GROUP"])
    Y_onehot = np.eye(2)[Y.astype(int)]
    np.save(X_PATH, X)           # (823, 264, 264)
    np.save(Y_PATH, Y_onehot)    # (823, 2)
    return processed_df, X, Y, Y_onehot


def split_traintest_sbj(Y, test_split_frac, seed):
    X = np.zeros(Y.shape[0])
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split_frac, random_state=seed
    )
    train_index, test_index = next(sss.split(X, Y))
    return train_index, test_index


def split_kfoldcv_sbj(Y, n, seed):
    X = np.zeros(Y.shape[0])
    skf_group = StratifiedKFold(
        n_splits=n, shuffle=True, random_state=seed
    )
    result = []
    for train_index, test_index in skf_group.split(X, Y):
        result.append((train_index, test_index))
    return result


def generate_splits(Y, test_split_frac=0.2, kfold_n_splits=5):
    """
    splits: np.ndarray with dimension 100 x 5 x 2
        - test indices of seed n = splits[n][0]
        - the train and val indices of seed n, fold k = splits[n][1][k][0] and splits[n][1][k][1]
    """
    splits = []
    for seed in range(100):
        np.random.seed(seed)
        tuning_idx, test_idx = split_traintest_sbj(Y, test_split_frac, seed)
        Y_tuning = Y[tuning_idx]
        folds = split_kfoldcv_sbj(Y_tuning, kfold_n_splits, seed)
        train_val_idx = []
        for tuning_train_idx, tuning_val_idx in folds:
            train_idx = tuning_idx[tuning_train_idx]
            val_idx = tuning_idx[tuning_val_idx]
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0
            train_val_idx.append(np.array([train_idx, val_idx], dtype=object))
        train_val_idx = np.array(train_val_idx)
        split = np.empty(2, dtype=object)
        split[0] = test_idx
        split[1] = train_val_idx
        splits.append(split)
    splits = np.array(splits, dtype=object)
    return splits


def generate_ssl_splits(Y, real_idx, test_split_frac=0.2, kfold_n_splits=5):
    """
    splits: np.ndarray with dimension 100 x 5 x 2
        - test indices of seed n = splits[n][0]
        - the train and val indices of seed n, fold k = splits[n][1][k][0] and splits[n][1][k][1]
    """
    warnings.filterwarnings("ignore")
    splits = []
    for seed in range(100):
        np.random.seed(seed)
        tuning_idx, test_idx = split_traintest_sbj(Y[real_idx], test_split_frac, seed)
        tuning_idx = real_idx[tuning_idx]
        test_idx = real_idx[test_idx]
        Y_tuning = Y[tuning_idx]
        folds = split_kfoldcv_sbj(Y_tuning, kfold_n_splits, seed)
        train_val_idx = []
        for tuning_train_idx, tuning_val_idx in folds:
            train_idx = tuning_idx[tuning_train_idx]
            val_idx = tuning_idx[tuning_val_idx]
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0
            train_val_idx.append(np.array([train_idx, val_idx], dtype=object))
        train_val_idx = np.array(train_val_idx)
        split = np.empty(2, dtype=object)
        split[0] = test_idx
        split[1] = train_val_idx
        splits.append(split)
    splits = np.array(splits, dtype=object)
    return splits


def split(Y):
    splits = generate_splits(Y)
    np.save(SPLITS_PATH, splits) # (100, 5, 2)


def ssl_split(meta_df, Y):
    for site_id in np.unique(meta_df['SITE_ID']):
        with log_time("generate ssl split seed for site {}".format(site_id)) as lt:
            try:
                site_idx = np.argwhere(meta_df['SITE_ID'].values == site_id).flatten()
                splits = generate_ssl_splits(Y, site_idx)
                np.save(os.path.join(SSL_SPLITS_DIR, "{}.npy".format(site_id)), splits) # (100, 5, 2)
            except Exception as e:
                if site_id != "CMU":
                    raise e
                print("SITE_ID: {}, ERROR: {}".format(site_id, e))


def corr_mx_flatten(X):
    """
    returns upper triangluar matrix of each sample in X
    X.shape == (num_sample, num_feature, num_feature)
    X_flattened.shape == (num_sample, num_feature * (num_feature - 1) / 2)
    """
    upper_triangular_idx = np.triu_indices(X.shape[1], 1)
    X_flattened = X[:, upper_triangular_idx[0], upper_triangular_idx[1]]
    return X_flattened


def combat_harmonization(X, meta_df):
    X = corr_mx_flatten(X)
    covars = meta_df[["SITE_ID", "AGE_AT_SCAN", "SEX"]]
    categorical_cols = ["SEX"]
    batch_col = "SITE_ID"
    combat = neuroCombat(
        dat=X.T, covars=covars, batch_col=batch_col,
        categorical_cols=categorical_cols
    )
    harmonized_X = combat["data"].T
    harmonized_X = np.array([squareform(x) for x in harmonized_X])
    np.save(HARMONIZED_X_PATH, harmonized_X)


if __name__ == "__main__":

    main_dir = "/data/data_repo/neuro_img/ABIDE"
    corr_mat_dir = os.path.join(main_dir, "fmri", "processed_corr_mat")
    meta_csv_path = os.path.join(main_dir, "meta", "Phenotypic_V1_0b_preprocessed1.csv")

    if not os.path.exists(SSL_SPLITS_DIR):
        os.makedirs(SSL_SPLITS_DIR)

    with log_time("extract metadata and correlation matrices") as lt:
        meta_df, X, Y, Y_onehot = extract_data(main_dir, corr_mat_dir, meta_csv_path)

    with log_time("generate split seed for whole dataset") as lt:
        split(Y)

    ssl_split(meta_df, Y)

    with log_time("neuroCombat") as lt:
        combat_harmonization(X, meta_df)