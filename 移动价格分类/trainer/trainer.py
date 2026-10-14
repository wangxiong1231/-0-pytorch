import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics.metric import Metrics
from utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion,
                 optimizer, scheduler, cfg, device, logger):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.logger = logger

        self.epochs = cfg["train"]["epochs"]
        self.grad_clip = cfg["train"]["grad_clip"]
        self.patience = cfg["train"]["early_stop_patience"]
        self.save_dir = Path(cfg["train"]["save_dir"]) / cfg["exp_name"]
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.save_dir / "tb"))
        self.best_metric = -1.0
        self.bad_epochs = 0
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"[Train] epoch {epoch}")
        for step, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

            if step % self.cfg["train"]["log_interval"] == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
            self.global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(1, n)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        metrics = Metrics(self.cfg["data"]["num_classes"])
        total_loss, n = 0.0, 0
        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            metrics.update(logits, y)
        return total_loss / max(1, n), metrics.compute()

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate()
            lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"acc={val_metrics['acc']:.4f} f1={val_metrics['f1_macro']:.4f} "
                f"auc={val_metrics['auc_macro']:.4f} lr={lr:.2e} "
                f"time={time.time()-t0:.1f}s"
            )

            self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("lr", lr, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            cur = val_metrics["acc"]
            if cur > self.best_metric:
                self.best_metric = cur
                self.bad_epochs = 0
                save_checkpoint(self.save_dir / "best.pt", self.model,
                                self.optimizer, self.scheduler, epoch,
                                self.best_metric, self.cfg)
                self.logger.info(f"  ↑ best acc={cur:.4f}, saved to best.pt")
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    self.logger.info(f"Early stop at epoch {epoch}.")
                    break

        self.writer.close()
        self.logger.info(f"Training done. Best acc = {self.best_metric:.4f}")