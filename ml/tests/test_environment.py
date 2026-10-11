import torch
import numpy as np
import pandas as pd
import sklearn


def test_ml_environment():
    assert torch.__version__
    assert np.__version__
    assert pd.__version__
    assert sklearn.__version__