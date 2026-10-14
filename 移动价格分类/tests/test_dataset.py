import pandas as pd
import torch

from data.dataset import TabularDataset


def test_tabular_dataset_with_label():
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "f2": [4.0, 5.0, 6.0],
        "label": [0, 1, 0],
    })
    ds = TabularDataset(df, feature_cols=["f1", "f2"], label_col="label")
    assert len(ds) == 3
    x, y = ds[0]
    assert x.shape == (2,)
    assert y.item() == 0
    assert ds.num_features == 2


def test_tabular_dataset_no_label():
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    ds = TabularDataset(df, feature_cols=["f1", "f2"], label_col=None)
    x = ds[0]
    assert x.shape == (2,)
    assert not isinstance(x, tuple)