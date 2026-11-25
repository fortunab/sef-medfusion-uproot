"""Kather sensitivity analysis holding out entire filename source families.

The 01..10 prefixes are inferred source families, not verified patient IDs.
This supplements the primary tile-level experiment without relabelling it.
"""

import argparse, json, re, time
from pathlib import Path
import numpy as np
import torch
from sef_medfusion.attacks import attack_batches
from sef_medfusion.data import prepare_data, load_data
from sef_medfusion.detector import collect_signals, calibrate, scores
from sef_medfusion.metrics import classification, detection
from sef_medfusion.models import CompactCNN
from sef_medfusion.training import train, predict
from sef_medfusion.utils import seed_all, sha256, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=Path("data"))
parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
parser.add_argument("--results-root", type=Path, default=Path("results"))
args = parser.parse_args()
config = json.loads((args.results_root / "config.json").read_text())
cache = prepare_data(
    "kather2016", args.data_root, config["image_size"], config["split_seed"]
)
x, y, _, manifest = load_data(cache)
groups = np.array(
    [re.search(r"CRC-Prim-HE-(\d{2})", r["id"]).group(1) for r in manifest["records"]]
)
unique = np.random.default_rng(config["split_seed"]).permutation(sorted(set(groups)))
assert len(unique) == 10
assignments = {
    group: split
    for subset, split in zip(
        [unique[:6], unique[6:7], unique[7:8], unique[8:]],
        ["train", "tune", "calib", "test"],
    )
    for group in subset
}
split = np.array([assignments[g] for g in groups])
folder = args.results_root / "kather_source_holdout"
folder.mkdir(parents=True, exist_ok=True)
write_json(
    folder / "manifest.json",
    {
        "group_meaning": "inferred filename source family; not verified patient",
        "assignments": assignments,
        "counts": {
            s: int((split == s).sum()) for s in ["train", "tune", "calib", "test"]
        },
        "records": [
            {
                "id": r["id"],
                "label": r["label"],
                "source_family": str(g),
                "split": str(s),
            }
            for r, g, s in zip(manifest["records"], groups, split)
        ],
    },
)
seed = config["seeds"][0]
seed_all(seed, config["threads"])
width = 8
model = CompactCNN(width, len(manifest["classes"]))
started = time.perf_counter()
info = train(
    model,
    x[split == "train"],
    y[split == "train"],
    x[split == "tune"],
    y[split == "tune"],
    config,
    lambda r: (
        print(
            "Source holdout epoch",
            r["epoch"],
            "tuneF1",
            round(r["tune_macro_f1"], 4),
            flush=True,
        )
        if r["epoch"] % 5 == 0
        else None
    ),
)
checkpoint = args.checkpoint_root / "kather_source_holdout" / "width_8.pt"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "state_dict": model.state_dict(),
        "width": width,
        "classes": len(manifest["classes"]),
        "class_names": manifest["classes"],
        "image_size": config["image_size"],
    },
    checkpoint,
)
test_x, test_y = x[split == "test"], y[split == "test"]
clean = collect_signals(model, test_x)
calibration = calibrate(
    collect_signals(model, x[split == "calib"]),
    config["alpha"],
    config["calibration_quantile"],
)
torch.manual_seed(seed + 208)
adversarial = attack_batches(
    model,
    test_x,
    test_y,
    batch_size=32,
    epsilon=8 / 255,
    steps=config["pgd_steps"],
    step_size=config["attack_step_size"],
)
assert float((adversarial - test_x).abs().max()) <= 8 / 255 + 1e-6
perturbed = collect_signals(model, adversarial)
clean_score = scores(clean, calibration.gradient_scale, calibration.alpha)["combined"]
adv_score = scores(perturbed, calibration.gradient_scale, calibration.alpha)["combined"]
success = (clean["probabilities"].argmax(1) == test_y) & (
    perturbed["probabilities"].argmax(1) != test_y
)
result = {
    "seed": seed,
    "width": width,
    "source_groups": assignments,
    "counts": {s: int((split == s).sum()) for s in ["train", "tune", "calib", "test"]},
    "classification": classification(test_y.numpy(), clean["probabilities"].numpy()),
    "pgd20_08": classification(test_y.numpy(), perturbed["probabilities"].numpy()),
    "combined_detector": detection(
        clean_score.numpy(),
        adv_score.numpy(),
        calibration.thresholds["combined"],
        success.numpy(),
    ),
    "max_pixel_delta": float((adversarial - test_x).abs().max()),
    "checkpoint_sha256": sha256(checkpoint),
    "elapsed_seconds": time.perf_counter() - started,
    **info,
}
np.savez_compressed(
    folder / "predictions.npz",
    labels=test_y.numpy(),
    probabilities=clean["probabilities"].numpy(),
    adversarial_probabilities=perturbed["probabilities"].numpy(),
    clean_score=clean_score.numpy(),
    adversarial_score=adv_score.numpy(),
)
write_json(folder / "calibration.json", calibration.to_dict())
write_json(folder / "results.json", result)
print(
    "SOURCE HOLDOUT",
    result["counts"],
    result["classification"],
    result["combined_detector"],
    flush=True,
)
