from __future__ import annotations
import torch
from torch import nn


class CompactCNN(nn.Module):
    """Three two-convolution blocks; no batch-dependent normalization.

    Pixel inputs are RGB float32 in [0,1]. Group normalization keeps detector
    input gradients independent of the other images in a minibatch.
    """

    def __init__(self, width: int, classes: int):
        super().__init__()
        blocks = []
        channels = 3
        for out in [width, 2 * width, 4 * width]:
            for _ in range(2):
                blocks += [
                    nn.Conv2d(channels, out, 3, padding=1, bias=False),
                    nn.GroupNorm(4, out),
                    nn.ReLU(),
                ]
                channels = out
            blocks += [nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def proxy_score(model: nn.Module, batches) -> dict:
    """Minimize parameters / mean global parameter-gradient L2 norm.

    This is the equation in the supplied manuscript, not a reproduction of
    the separate activation-diversity AD-ZeroNAS publication.
    """
    model.eval()
    norms = []
    for images, labels in batches:
        logits = model(images)
        grads = torch.autograd.grad(
            nn.functional.cross_entropy(logits, labels), model.parameters()
        )
        norm = torch.sqrt(sum(g.square().sum() for g in grads))
        norms.append(float(norm))
    mean = sum(norms) / len(norms)
    count = parameter_count(model)
    return {
        "parameters": count,
        "gradient_norms": norms,
        "mean_gradient_norm": mean,
        "score": count / max(mean, 1e-12),
    }
