import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


class Metrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.probs = []
        self.targets = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.softmax(logits.detach(), dim=-1)
        self.probs.append(probs.cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> dict[str, float]:
        probs = torch.cat(self.probs).numpy()
        targets = torch.cat(self.targets).numpy()
        preds = probs.argmax(axis=1)

        acc = float((preds == targets).mean())
        f1_macro = float(f1_score(targets, preds, average="macro", zero_division=0))
        try:
            if self.num_classes > 2:
                auc_macro = float(roc_auc_score(
                    targets, probs, multi_class="ovr", average="macro"))
            else:
                auc_macro = float(roc_auc_score(targets, probs[:, 1]))
        except ValueError:
            auc_macro = float("nan")
        return {"acc": acc, "f1_macro": f1_macro, "auc_macro": auc_macro}