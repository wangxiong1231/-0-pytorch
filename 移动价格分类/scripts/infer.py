import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from data.dataloader import build_dataloaders, FEATURE_COLS
from data.dataset import TabularDataset
from models.model import build_model
from utils.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/pred.csv")
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    _, _, _, scaler, in_dim = build_dataloaders(cfg)
    cfg["data"]["in_dim"] = in_dim

    test_df = pd.read_csv(cfg["data"]["test_csv"])
    ds = TabularDataset(test_df, FEATURE_COLS, label_col=None, scaler=scaler)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False)

    model = build_model(cfg)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.to(device).eval()

    all_preds, all_probs = [], []
    for x in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=-1).cpu()
        all_probs.append(probs)
        all_preds.append(probs.argmax(dim=-1))

    preds = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()

    out = pd.DataFrame({"id": test_df["id"], "pred": preds})
    for c in range(cfg["data"]["num_classes"]):
        out[f"prob_{c}"] = probs[:, c]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} predictions to {args.out}")
    print(out.head())


if __name__ == "__main__":
    main()