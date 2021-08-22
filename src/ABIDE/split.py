import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from abide_config import MAIN_DIR, Y_FNAME, SPLITS_FNAME


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
        splits.append(np.array([test_idx, np.array(train_val_idx)], dtype=object))
    splits = np.array(splits, dtype=object)
    return splits


def main(y_path, output_dir):
    Y_onehot = np.load(y_path)
    Y = np.argmax(Y_onehot, axis=1)
    splits = generate_splits(Y)
    np.save(os.path.join(output_dir, SPLITS_FNAME), splits) # (100, 5, 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help="extracted Y.npy path", 
                        default=os.path.join(MAIN_DIR, Y_FNAME))
    parser.add_argument('--output', '-o', type=str, help="output directory", 
                        default=MAIN_DIR)
    args = parser.parse_args()

    y_path = os.path.abspath(args.input)
    if not os.path.exists(y_path):
        raise FileNotFoundError("{} does not exists".format(y_path))

    output_dir = os.path.abspath(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(y_path, output_dir)