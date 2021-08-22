import os

__dir__ = os.path.dirname(__file__)


META_CSV_FNAME = "meta.csv"
X_FNAME = "X.npy"
Y_FNAME = "Y.npy"
SPLITS_FNAME = "splits.npy"
SSL_SPLITS_FNAME = "ssl_splits.npy"

SSL_SITE_ID = "NYU"
MAIN_DIR = os.path.abspath(os.path.join(
    __dir__, "..", "..", "data", "ABIDE"
))
META_CSV_PATH = os.path.join(MAIN_DIR, META_CSV_FNAME)