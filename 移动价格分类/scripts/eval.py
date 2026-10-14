import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from utils.config import load_config
from utils.seed import set_seed
from data.dataloader import build_dataloaders
from models.model import build_model
from losses.loss import build_loss
from metrics.metric import Metrics
from utils.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    metrics = Metrics(num_classes)
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        metrics.update(logits, y)
    return total_loss / max(1, n), metrics.compute()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    _, val_loader, _, _, in_dim = build_dataloaders(cfg)
    cfg["data"]["in_dim"] = in_dim

    model = build_model(cfg)
    criterion = build_loss(cfg)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.to(device)

    loss, m = evaluate(model, val_loader, criterion, device, cfg["data"]["num_classes"])
    print(f"[VAL] loss={loss:.4f} acc={m['acc']:.4f} f1={m['f1_macro']:.4f} auc={m['auc_macro']:.4f}")


if __name__ == "__main__":
    main()