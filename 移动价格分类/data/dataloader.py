from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data.dataset import TabularDataset


# 显式特征列，排除 id 和 label
FEATURE_COLS = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]


def build_dataloaders(cfg: dict):
    seed = cfg["seed"]
    label_col = cfg["data"]["label_col"]

    train_df = pd.read_csv(cfg["data"]["train_csv"])
    assert label_col in train_df.columns, f"train.csv 缺少标签列 {label_col}"

    train_split, val_split = train_test_split(
        train_df,
        test_size=cfg["data"]["val_ratio"],
        stratify=train_df[label_col],
        random_state=seed,
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(train_split[FEATURE_COLS].values)

    out_dir = Path(cfg["train"]["save_dir"]) / cfg["exp_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_split.to_csv(out_dir / "train_split.csv", index=False)
    val_split.to_csv(out_dir / "val_split.csv", index=False)

    ds_train = TabularDataset(train_split, FEATURE_COLS, label_col, scaler)
    ds_val = TabularDataset(val_split, FEATURE_COLS, label_col, scaler)

    ds_test = None
    test_csv = cfg["data"].get("test_csv")
    if test_csv and Path(test_csv).exists():
        test_df = pd.read_csv(test_csv)
        ds_test = TabularDataset(test_df, FEATURE_COLS, label_col=None, scaler=scaler)

    common = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    train_loader = DataLoader(ds_train, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(ds_val, shuffle=False, **common)
    test_loader = DataLoader(ds_test, shuffle=False, **common) if ds_test else None

    return train_loader, val_loader, test_loader, scaler, ds_train.num_features