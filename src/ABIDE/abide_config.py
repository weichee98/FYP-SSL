import os


__dir__ = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


MAIN_DIR = os.path.abspath(os.path.join(__dir__, "data", "ABIDE"))
META_CSV_PATH = os.path.join(MAIN_DIR, "meta.csv")
X_PATH = os.path.join(MAIN_DIR, "X.npy")
Y_PATH = os.path.join(MAIN_DIR, "Y.npy")
SPLIT_TEST_PATH = os.path.join(MAIN_DIR, "split_test.npy")
SPLIT_CV_PATH = os.path.join(MAIN_DIR, "split_cv.npy")
SSL_SPLITS_DIR = os.path.join(MAIN_DIR, "ssl_splits")
HARMONIZED_X_PATH = os.path.join(MAIN_DIR, "harmonized_X.npy")