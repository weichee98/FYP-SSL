import os
import argparse
import numpy as np
import pandas as pd
from abide_config import MAIN_DIR, META_CSV_FNAME, X_FNAME, Y_FNAME


main_dir = "/data/data_repo/neuro_img/ABIDE"
corr_mat_dir = os.path.join(main_dir, "fmri", "processed_corr_mat")
meta_csv_path = os.path.join(main_dir, "meta", "Phenotypic_V1_0b_preprocessed1.csv")


def get_processed_corr_mat_file_ids():
    file_ids = []
    for dir, _, files in os.walk(corr_mat_dir):
        for filename in files:
            if filename.endswith(".npy"):
                file_ids.append(filename[:-10])
    return file_ids


def get_file_path(dx_group, file_id):
    filename = "{}_power.npy".format(file_id)
    if dx_group == 1:
        return os.path.join(corr_mat_dir, "diseased", filename)
    else:
        return os.path.join(corr_mat_dir, "normal", filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, help="output directory", 
                        default=MAIN_DIR)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    meta_df = pd.read_csv(meta_csv_path, index_col=0)
    meta_df = meta_df.drop(["Unnamed: 0.1", "subject"], axis=1)
    file_ids = get_processed_corr_mat_file_ids()

    meta_df["PROCESSED"] = meta_df["FILE_ID"].apply(lambda x: x in file_ids)
    processed_df = meta_df[meta_df["PROCESSED"]].sort_values("SUB_ID")
    processed_df = processed_df.drop("PROCESSED", axis=1)
    processed_df["FILE_PATH"] = processed_df[["DX_GROUP", "FILE_ID"]].apply(
        lambda x: get_file_path(x["DX_GROUP"], x["FILE_ID"]), axis=1
    )

    output_path = os.path.join(output_dir, META_CSV_FNAME)
    processed_df.to_csv(output_path, header=True, index=False)

    X = np.array([np.load(fname) for fname in processed_df["FILE_PATH"]])
    X = np.nan_to_num(X)
    Y = np.array(2 - processed_df["DX_GROUP"])
    Y_onehot = np.eye(2)[Y.astype(int)]
    np.save(os.path.join(output_dir, X_FNAME), X)           # (823, 264, 264)
    np.save(os.path.join(output_dir, Y_FNAME), Y_onehot)    # (823, 2)