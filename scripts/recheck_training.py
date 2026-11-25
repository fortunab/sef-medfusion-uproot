"""Independently retrain the first-seed proxy winner and compare exact weights."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sef_medfusion.data import prepare_data, load_data
from sef_medfusion.experiment import load_model
from sef_medfusion.models import CompactCNN
from sef_medfusion.training import train, predict
from sef_medfusion.utils import seed_all, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=Path("data"))
parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
parser.add_argument("--results-root", type=Path, default=Path("results"))
parser.add_argument("--dataset", action="append")
args = parser.parse_args()
config = json.loads((args.results_root / "config.json").read_text())
for dataset in args.dataset or config["datasets"]:
    started = time.perf_counter()
    seed = config["seeds"][0]
    folder = args.results_root / dataset / f"seed_{seed}"
    width = json.loads((folder / "proxy.json").read_text())["selected_width"]
    cache = prepare_data(
        dataset, args.data_root, config["image_size"], config["split_seed"]
    )
    x, y, split, manifest = load_data(cache)
    seed_all(seed, config["threads"])
    fresh = CompactCNN(width, len(manifest["classes"]))
    info = train(
        fresh,
        x[split == "train"],
        y[split == "train"],
        x[split == "tune"],
        y[split == "tune"],
        config,
        lambda row: (
            print(
                dataset,
                "recheck epoch",
                row["epoch"],
                "F1",
                round(row["tune_macro_f1"], 4),
                flush=True,
            )
            if row["epoch"] % 5 == 0
            else None
        ),
    )
    previous, _ = load_model(
        args.checkpoint_root / dataset / f"seed_{seed}" / f"width_{width}.pt"
    )
    deltas = {
        key: float((value - previous.state_dict()[key]).abs().max())
        for key, value in fresh.state_dict().items()
    }
    with np.load(folder / f"width_{width}_predictions.npz", allow_pickle=False) as z:
        difference = float(
            np.abs(predict(fresh, x[split == "test"]) - z["probabilities"]).max()
        )
    result = {
        "dataset": dataset,
        "seed": seed,
        "width": width,
        "max_weight_difference": max(deltas.values()),
        "max_probability_difference": difference,
        "best_epoch": info["best_epoch"],
        "elapsed_seconds": time.perf_counter() - started,
        "bitwise_equal": max(deltas.values()) == 0 and difference == 0,
    }
    write_json(folder / "reproduction_check.json", result)
    assert result["bitwise_equal"], result
    print("REPRODUCED", result, flush=True)
