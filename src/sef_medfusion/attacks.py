from __future__ import annotations
import torch
from torch.nn import functional as F
from .detector import input_signals, scores


def attack(
    model,
    images,
    labels,
    epsilon,
    steps=20,
    step_size=2 / 255,
    random_start=True,
    calibration=None,
    adaptive_lambda=1.0,
):
    """Untargeted L-infinity PGD, or FGSM with steps=1/random_start=False.

    Optional adaptive objective: CE(true label) - lambda * detector score.
    Input-gradient differentiation uses create_graph=True for this objective.
    It is one detector-aware stress test, not an exhaustive security audit.
    """
    model.eval()
    origin = images.detach()
    x = origin.clone()
    if random_start:
        x = (x + torch.empty_like(x).uniform_(-epsilon, epsilon)).clamp(0, 1)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        if calibration is None:
            objective = F.cross_entropy(model(x), labels, reduction="sum")
        else:
            signal = input_signals(model, x, create_graph=True)
            score = scores(signal, calibration.gradient_scale, calibration.alpha)[
                "combined"
            ]
            objective = (
                F.cross_entropy(signal["logits"], labels, reduction="none")
                - adaptive_lambda * score
            ).sum()
        gradient = torch.autograd.grad(objective, x)[0]
        x = x.detach() + step_size * gradient.sign()
        x = torch.maximum(torch.minimum(x, origin + epsilon), origin - epsilon).clamp(
            0, 1
        )
    return x.detach()


def attack_batches(model, images, labels, batch_size=64, **kwargs):
    return torch.cat(
        [
            attack(model, x, y, **kwargs)
            for x, y in zip(images.split(batch_size), labels.split(batch_size))
        ]
    )
