import os
import argparse
import numpy as np
import pandas as pd
from split import split_traintest_sbj, split_kfoldcv_sbj
from abide_config import MAIN_DIR, META_CSV_PATH, SSL_SPLITS_FNAME, SSL_SITE_ID


def generate_splits(Y, real_idx, test_split_frac=0.2, kfold_n_splits=5):
    """
    splits: np.ndarray with dimension 100 x 5 x 2
        - test indices of seed n = splits[n][0]
        - the train and val indices of seed n, fold k = splits[n][1][k][0] and splits[n][1][k][1]
    """
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
        splits.append(np.array([test_idx, np.array(train_val_idx)], dtype=object))
    splits = np.array(splits, dtype=object)
    return splits


def main(meta_csv_path, output_dir, site_id):
    meta_df = pd.read_csv(meta_csv_path, index_col=0)
    Y = np.array(2 - meta_df["DX_GROUP"])
    site_idx = np.argwhere(meta_df['SITE_ID'].values == site_id).flatten()
    splits = generate_splits(Y, site_idx)
    np.save(os.path.join(output_dir, SSL_SPLITS_FNAME), splits) # (100, 5, 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help="input metadata csv path", 
                        default=META_CSV_PATH)
    parser.add_argument('--target', '-t', type=str, help="the target SITE_ID", 
                        default=SSL_SITE_ID)
    parser.add_argument('--output', '-o', type=str, help="output directory", 
                        default=MAIN_DIR)
    args = parser.parse_args()

    meta_csv_path = os.path.abspath(args.input)
    if not os.path.exists(meta_csv_path):
        raise FileNotFoundError("{} does not exists".format(meta_csv_path))

    output_dir = os.path.abspath(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(meta_csv_path, output_dir, args.target)