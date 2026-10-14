import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import math
import torch

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import build_logger
from data.dataloader import build_dataloaders
from models.model import build_model
from losses.loss import build_loss
from trainer.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("overrides", nargs="*", help="key=value 例如 optimizer.lr=1e-4")
    return p.parse_args()


def build_optimizer(model, cfg):
    name = cfg["optimizer"]["name"].lower()
    lr = cfg["optimizer"]["lr"]
    wd = cfg["optimizer"]["weight_decay"]

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg):
    name = cfg["scheduler"]["name"].lower()
    epochs = cfg["train"]["epochs"]
    warmup = cfg["scheduler"]["warmup_epochs"]
    min_lr = cfg["scheduler"]["min_lr"]
    base_lr = cfg["optimizer"]["lr"]

    if name == "cosine":
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = (epoch - warmup) / max(1, epochs - warmup)
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cos)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    save_dir = Path(cfg["train"]["save_dir"]) / cfg["exp_name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(save_dir)
    logger.info(f"Experiment: {cfg['exp_name']}")

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader, val_loader, test_loader, scaler, in_dim = build_dataloaders(cfg)
    cfg["data"]["in_dim"] = in_dim

    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Config:\n{json.dumps(cfg, indent=2)}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    logger.info(f"Model:\n{model}")

    trainer = Trainer(model, train_loader, val_loader, criterion,
                      optimizer, scheduler, cfg, device, logger)
    trainer.fit()


if __name__ == "__main__":
    main()