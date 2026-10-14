import torch
import torch.nn as nn


def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(labels), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


def build_loss(cfg: dict, class_weights: torch.Tensor | None = None) -> nn.Module:
    name = cfg["loss"]["name"].lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg["loss"].get("label_smoothing", 0.0),
        )
    raise ValueError(f"Unknown loss: {name}")