import torch

from models.model import build_model


def test_mlp_forward():
    cfg = {
        "model": {"name": "mlp", "hidden_dims": [128, 64], "dropout": 0.1},
        "data": {"num_classes": 4, "in_dim": 20},
    }
    model = build_model(cfg).eval()
    x = torch.randn(8, 20)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8, 4)