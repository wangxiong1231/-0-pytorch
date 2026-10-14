import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"]["name"].lower()
    if name != "mlp":
        raise ValueError(f"Unknown model: {name}")

    in_dim = cfg["data"]["in_dim"]
    num_classes = cfg["data"]["num_classes"]
    hidden = cfg["model"]["hidden_dims"]
    dropout = cfg["model"]["dropout"]

    layers = []
    prev = in_dim
    for h in hidden:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)