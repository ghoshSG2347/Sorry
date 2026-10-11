import numpy as np


def normalize_features(X):
    """
    Convert features to float32 and normalize them
    to the range [0, 1] when possible.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.max() > 1.0:
        X = X / 255.0

    return X