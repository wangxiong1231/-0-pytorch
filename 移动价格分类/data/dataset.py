from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """纯表格数据集。label_col=None 时返回单个 tensor（用于 test / infer）。"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_cols: list[str],
        label_col: str | None = "price_range",
        scaler=None,
    ) -> None:
        self.feature_cols = feature_cols
        self.label_col = label_col

        x = dataframe[feature_cols].values.astype(np.float32)
        if scaler is not None:
            x = scaler.transform(x).astype(np.float32)
        self.x = torch.from_numpy(x)

        if label_col is not None and label_col in dataframe.columns:
            self.y = torch.from_numpy(
                dataframe[label_col].values.astype(np.int64))
        else:
            self.y = None

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]