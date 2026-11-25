from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    pred = p.argmax(1)
    confidence = p.max(1)
    correct = pred == y
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(
                y, pred, labels=np.arange(p.shape[1]), average="macro", zero_division=0
            )
        ),
        "confidence": float(confidence.mean()),
        "ece_15": float(ece),
        "nll": float(-np.log(p[np.arange(len(y)), y].clip(1e-12, 1)).mean()),
        "brier": float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()),
        "n": int(len(y)),
    }


def detection(clean_scores, perturbed_scores, threshold, success_mask):
    clean, adv = np.asarray(clean_scores), np.asarray(perturbed_scores)
    success = np.asarray(success_mask, dtype=bool)
    auc = roc_auc_score(
        np.r_[np.zeros(len(clean)), np.ones(len(adv))], np.r_[clean, adv]
    )
    return {
        "auroc": float(auc),
        "fpr": float((clean > threshold).mean()),
        "tpr_all": float((adv > threshold).mean()),
        "tpr_successful": (
            float((adv[success] > threshold).mean()) if success.any() else None
        ),
        "successful_attacks": int(success.sum()),
        "n": int(len(adv)),
        "threshold": float(threshold),
    }
