from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch.nn import functional as F


def input_signals(model, images: torch.Tensor, create_graph: bool = False):
    """Inference-time pseudo-label CE, summed to preserve per-image scale."""
    x = images if images.requires_grad else images.detach().clone().requires_grad_(True)
    logits = model(x)
    probs = logits.softmax(1)
    pseudo = logits.detach().argmax(1)
    grad = torch.autograd.grad(
        F.cross_entropy(logits, pseudo, reduction="sum"), x, create_graph=create_graph
    )[0]
    norm = grad.flatten(1).norm(p=2, dim=1)
    confidence = probs.max(1).values
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(1)
    return {
        "logits": logits,
        "probabilities": probs,
        "gradient": norm,
        "confidence": confidence,
        "entropy": entropy,
    }


@dataclass
class Calibration:
    gradient_scale: float
    alpha: float
    quantile: float
    thresholds: dict[str, float]

    def to_dict(self):
        return asdict(self)


def scores(signals, scale: float, alpha: float = 0.5):
    gradient = (signals["gradient"] / max(scale, 1e-12)).clamp(0, 1)
    uncertainty = 1 - signals["confidence"]
    return {
        "combined": alpha * gradient + (1 - alpha) * uncertainty,
        "gradient": gradient,
        "uncertainty": uncertainty,
        "entropy": signals["entropy"],
    }


def collect_signals(model, images: torch.Tensor, batch_size: int = 64):
    model.eval()
    output = {key: [] for key in ["probabilities", "gradient", "confidence", "entropy"]}
    for x in images.split(batch_size):
        with torch.enable_grad():
            signal = input_signals(model, x)
        for key in output:
            output[key].append(signal[key].detach().cpu())
    return {key: torch.cat(value) for key, value in output.items()}


def calibrate(signal, alpha: float = 0.5, quantile: float = 0.95) -> Calibration:
    scale = max(float(signal["gradient"].max()), 1e-12)
    values = scores(signal, scale, alpha)
    thresholds = {
        k: float(np.quantile(v.numpy(), quantile, method="higher"))
        for k, v in values.items()
    }
    return Calibration(scale, alpha, quantile, thresholds)
