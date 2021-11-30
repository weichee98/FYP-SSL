import os
import time
import warnings
import numpy as np
import pandas as pd
from adhd_config import *
from contextlib import contextmanager
from neuroCombat import neuroCombat
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from joblib import Parallel, delayed


@contextmanager
def log_time(description):
    print("[{}] started".format(description))
    start = time.time()
    yield
    end = time.time()
    print("[{}] completed in {:.3f} s".format(description, end - start))


class ExtractData:

    def __init__(self, phenotypics_path, corr_mat_path) -> None:
        self.phenotypics_path = phenotypics_path
        self.corr_mat_path = corr_mat_path
        self.file_names = [x for x in os.listdir(corr_mat_path) if x.endswith(".npy")]

    @staticmethod
    def _standard_preprocess(meta_csv_path):
        df = pd.read_csv(meta_csv_path)
        df.columns = list(map(lambda x: "_".join(x.split()).upper(), df.columns))
        return df

    @staticmethod
    def _get_file_id(df):
        return df["SCANDIR_ID"].apply(lambda x: int(str(x)[1:]))

    def _merge_df(self, df: pd.DataFrame, prefix: str):
        df["FILE_ID"] = df["FILE_ID"].apply(lambda x: prefix + "{:06d}".format(x))

        files = [x for x in self.file_names if x.startswith(prefix)]
        merge_df = pd.DataFrame({
            "FILE_ID": map(lambda x: x.split("_")[0], files),
            "FILE_PATH": map(lambda x: os.path.join(corr_mat_dir, x), files)
        })
        merge_df = merge_df.groupby("FILE_ID")["FILE_PATH"].apply(list).reset_index()
        df = df.merge(merge_df, on=["FILE_ID"])
        return df

    def _KKI(self):
        folder_name = "KKI_preproc_filtfix"
        meta_csv_path = os.path.join(self.phenotypics_path, folder_name, "KKI_phenotypic.csv")
        df = self._standard_preprocess(meta_csv_path)
        df["SITE_NAME"] = "KKI"
        df["FILE_ID"] = self._get_file_id(df)
        df = self._merge_df(df, "KKI-")
        return df

    def _NI(self):
        folder_name = "NeuroIMAGE"
        meta_csv_path = os.path.join(self.phenotypics_path, folder_name, "NeuroIMAGE_phenotypic.csv")
        df = self._standard_preprocess(meta_csv_path)
        df["SITE_NAME"] = "NI"
        df["FILE_ID"] = self._get_file_id(df)
        df = self._merge_df(df, "NeuroIMAGE-")
        return df

    def _NYU(self):
        folder_name = "NYU_preproc_filtfix"
        meta_csv_path = os.path.join(self.phenotypics_path, folder_name, "NYU_phenotypic.csv")
        df = self._standard_preprocess(meta_csv_path)
        df["SITE_NAME"] = "NYU"
        df["FILE_ID"] = self._get_file_id(df)
        df["FILE_ID"] = df["FILE_ID"].apply(lambda x: x + 10000 if x <= 129 else x)
        df = self._merge_df(df, "NYU-")
        return df

    def _OHSU(self):
        folder_name = "OHSU_preproc_filtfix"
        meta_csv_path = os.path.join(self.phenotypics_path, folder_name, "OHSU_phenotypic.csv")
        df = self._standard_preprocess(meta_csv_path)
        df["SITE_NAME"] = "OHSU"
        df["FILE_ID"] = self._get_file_id(df)
        df = self._merge_df(df, "OHSU-")
        return df

    def _PKU(self):
        folder_name = "Peking_preproc_filtfix"
        meta_csv_paths = [
            os.path.join(self.phenotypics_path, folder_name, "Peking_1_phenotypic.csv"),
            os.path.join(self.phenotypics_path, folder_name, "Peking_2_phenotypic.csv"),
            os.path.join(self.phenotypics_path, folder_name, "Peking_3_phenotypic.csv"),
        ]
        df = [self._standard_preprocess(path) for path in meta_csv_paths]
        df = pd.concat(df)
        df["SITE_NAME"] = "PKU"
        df["FILE_ID"] = self._get_file_id(df)
        df = self._merge_df(df, "Peking-")
        return df

    def _PITT(self):
        folder_name = "Pittsburgh"
        meta_csv_path = os.path.join(self.phenotypics_path, folder_name, "Pittsburgh_phenotypic.csv")
        df = self._standard_preprocess(meta_csv_path)
        df["SITE_NAME"] = "PITT"
        df["FILE_ID"] = df["SCANDIR_ID"]
        df = self._merge_df(df, "Pittsburgh-")
        return df

    def _WUSTL(self):
        folder_name = "WashU"
        meta_csv_path = os.path.join(self.phenotypics_path, folder_name, "WashU_phenotypic.csv")
        df = self._standard_preprocess(meta_csv_path)
        df = df.rename(columns={"SCANDIRID": "SCANDIR_ID"})
        df["SITE_NAME"] = "WUSTL"
        df["FILE_ID"] = df["SCANDIR_ID"]
        df = self._merge_df(df, "WashU-")
        return df

    def extract_data(self):
        dfs = [self._KKI, self._NI, self._NYU, self._OHSU, self._PKU, self._PITT, self._WUSTL]
        dfs = Parallel(n_jobs=7)(delayed(x)() for x in dfs)
        meta_df = pd.concat(dfs, ignore_index=True)
        meta_df = meta_df.sort_values(by=["FILE_ID"])
        return meta_df


def get_corr_mat(file_paths):
    X = np.array([np.load(fname) for fname in file_paths])
    if len(X) == 1:
        return X[0]
    X = np.arctanh(X)
    X = np.mean(X, axis=0)
    X = np.tanh(X)
    return X


def extract_data(corr_mat_dir, phenotypics_path):    
    ed = ExtractData(phenotypics_path, corr_mat_dir)
    processed_df = ed.extract_data()
    processed_df.to_csv(META_CSV_PATH, header=True, index=False)

    X = Parallel(n_jobs=10)(delayed(get_corr_mat)(fname) for fname in processed_df["FILE_PATH"])
    X = np.array(X)
    X = np.nan_to_num(X)
    Y = np.array(processed_df["DX"] != 0)
    Y_onehot = np.eye(2)[Y.astype(int)]
    np.save(X_PATH, X)          # (823, 264, 264)
    np.save(Y_PATH, Y_onehot)   # (823, 2)
    return processed_df, X


def split_traintest_sbj(df: pd.DataFrame, test_split_frac, seed):
    df_unique = df.drop_duplicates(subset="FILE_ID")
    file_ids = np.array(df_unique["FILE_ID"])
    Y = np.array(df_unique["DX"] != 0).astype(int)
    X = np.zeros(Y.shape[0])

    np.random.seed(seed)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split_frac, random_state=seed
    )
    train_index, test_index = next(sss.split(X, Y))
    train_files, test_files = file_ids[train_index], file_ids[test_index]

    all_files = np.array(df["FILE_ID"])
    train_index = np.argwhere(np.isin(all_files, train_files)).flatten()
    test_index = np.argwhere(np.isin(all_files, test_files)).flatten()

    assert len(np.intersect1d(all_files[train_index], all_files[test_index])) == 0
    return train_index, test_index


def split_kfoldcv_sbj(df: pd.DataFrame, n, seed):
    all_files = np.array(df["FILE_ID"])
    df_unique = df.drop_duplicates(subset="FILE_ID")
    file_ids = np.array(df_unique["FILE_ID"])
    Y = np.array(df_unique["DX"] != 0).astype(int)
    X = np.zeros(Y.shape[0])

    np.random.seed(seed)
    skf_group = StratifiedKFold(
        n_splits=n, shuffle=True, random_state=seed
    )
    result = []
    for train_index, test_index in skf_group.split(X, Y):
        train_files, test_files = file_ids[train_index], file_ids[test_index]
        train_index = np.argwhere(np.isin(all_files, train_files)).flatten()
        test_index = np.argwhere(np.isin(all_files, test_files)).flatten()
        assert len(np.intersect1d(all_files[train_index], all_files[test_index])) == 0
        result.append((train_index, test_index))
    return result


def generate_splits(df, test_split_frac=0.2, kfold_n_splits=5, test=True):
    """
    splits: np.ndarray with dimension 100 x 5 x 2
        - test indices of seed n = splits[n][0]
        - the train and val indices of seed n, fold k = splits[n][1][k][0] and splits[n][1][k][1]
    """
    splits = []
    for seed in range(100):
        if test:
            tuning_idx, test_idx = split_traintest_sbj(df, test_split_frac, seed)
        else:
            tuning_idx, test_idx = np.arange(df.shape[0]), np.array([])
        df_tuning = df.iloc[tuning_idx]
        folds = split_kfoldcv_sbj(df_tuning, kfold_n_splits, seed)
        train_val_idx = []
        for tuning_train_idx, tuning_val_idx in folds:
            train_idx = tuning_idx[tuning_train_idx]
            val_idx = tuning_idx[tuning_val_idx]
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0
            assert len(set(df["FILE_ID"].iloc[train_idx]) & set(df["FILE_ID"].iloc[val_idx])) == 0
            assert len(set(df["FILE_ID"].iloc[train_idx]) & set(df["FILE_ID"].iloc[test_idx])) == 0
            assert len(set(df["FILE_ID"].iloc[val_idx]) & set(df["FILE_ID"].iloc[test_idx])) == 0
            train_val_idx.append(np.array([train_idx, val_idx], dtype=object))
        train_val_idx = np.array(train_val_idx)
        split = np.empty(2, dtype=object)
        split[0] = test_idx
        split[1] = train_val_idx
        splits.append(split)
    splits = np.array(splits, dtype=object)
    return splits


def generate_ssl_splits(df, real_idx, test_split_frac=0.2, kfold_n_splits=5, test=True):
    """
    splits: np.ndarray with dimension 100 x 5 x 2
        - test indices of seed n = splits[n][0]
        - the train and val indices of seed n, fold k = splits[n][1][k][0] and splits[n][1][k][1]
    """
    warnings.filterwarnings("ignore")
    splits = []
    for seed in range(100):
        if test:
            tuning_idx, test_idx = split_traintest_sbj(df.iloc[real_idx], test_split_frac, seed)
            tuning_idx = real_idx[tuning_idx]
            test_idx = real_idx[test_idx]
        else:
            tuning_idx = real_idx
            test_idx = np.array([])
        df_tuning = df.iloc[tuning_idx]
        folds = split_kfoldcv_sbj(df_tuning, kfold_n_splits, seed)
        train_val_idx = []
        for tuning_train_idx, tuning_val_idx in folds:
            train_idx = tuning_idx[tuning_train_idx]
            val_idx = tuning_idx[tuning_val_idx]
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0
            assert len(set(df["FILE_ID"].iloc[train_idx]) & set(df["FILE_ID"].iloc[val_idx])) == 0
            assert len(set(df["FILE_ID"].iloc[train_idx]) & set(df["FILE_ID"].iloc[test_idx])) == 0
            assert len(set(df["FILE_ID"].iloc[val_idx]) & set(df["FILE_ID"].iloc[test_idx])) == 0
            train_val_idx.append(np.array([train_idx, val_idx], dtype=object))
        train_val_idx = np.array(train_val_idx)
        split = np.empty(2, dtype=object)
        split[0] = test_idx
        split[1] = train_val_idx
        splits.append(split)
    splits = np.array(splits, dtype=object)
    return splits


def split(meta_df):
    splits = generate_splits(meta_df, test=True)
    np.save(SPLIT_TEST_PATH, splits) # (100, 5, 2)
    splits = generate_splits(meta_df, test=False)
    np.save(SPLIT_CV_PATH, splits) # (100, 5, 2)


def ssl_split(meta_df):
    for site_id in np.unique(meta_df['SITE_NAME']):
        with log_time("generate ssl split seed for site {}".format(site_id)) as lt:
            try:
                site_idx = np.argwhere(meta_df['SITE_NAME'].values == site_id).flatten()
                splits = generate_ssl_splits(meta_df, site_idx, test=True)
                np.save(os.path.join(SSL_SPLITS_DIR, "{}_test.npy".format(site_id)), splits) # (100, 5, 2)
                splits = generate_ssl_splits(meta_df, site_idx, test=False)
                np.save(os.path.join(SSL_SPLITS_DIR, "{}_cv.npy".format(site_id)), splits) # (100, 5, 2)
            except Exception as e:
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
    covars = meta_df[["SITE", "AGE", "GENDER"]]
    categorical_cols = ["GENDER"]
    continuous_cols = ["AGE"]
    batch_col = "SITE"
    combat = neuroCombat(
        dat=X.T, covars=covars, batch_col=batch_col,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
    )
    harmonized_X = combat["data"].T
    harmonized_X = np.clip(harmonized_X, -1, 1)
    harmonized_X = np.array([squareform(x) for x in harmonized_X])
    np.save(HARMONIZED_X_PATH, harmonized_X)


if __name__ == "__main__":

    main_dir = "/data/data_repo/neuro_img/ADHD-200"
    corr_mat_dir = os.path.join(main_dir, "fmri", "processed_corr_mat")
    phenotypics_path = os.path.join(main_dir, "fmri", "raw")

    if not os.path.exists(SSL_SPLITS_DIR):
        os.makedirs(SSL_SPLITS_DIR)

    with log_time("extract metadata and correlation matrices") as lt:
        meta_df, X = extract_data(corr_mat_dir, phenotypics_path)

    with log_time("generate split seed for whole dataset") as lt:
        split(meta_df)

    ssl_split(meta_df)

    with log_time("neuroCombat") as lt:
        combat_harmonization(X, meta_df)