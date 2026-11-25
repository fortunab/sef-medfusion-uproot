"""Repeat timing after training/attack jobs finish to avoid own-job contention."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sef_medfusion.data import prepare_data, load_data
from sef_medfusion.detector import input_signals
from sef_medfusion.experiment import load_model
from sef_medfusion.utils import seed_all, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=Path("data"))
parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
parser.add_argument("--results-root", type=Path, default=Path("results"))
args = parser.parse_args()
config = json.loads((args.results_root / "config.json").read_text())
for dataset in config["datasets"]:
    cache = prepare_data(
        dataset, args.data_root, config["image_size"], config["split_seed"]
    )
    x, _, split, _ = load_data(cache)
    image = x[split == "test"][:1]
    for seed in config["seeds"]:
        seed_all(seed, config["threads"])
        folder = args.results_root / dataset / f"seed_{seed}"
        width = json.loads((folder / "proxy.json").read_text())["selected_width"]
        model, _ = load_model(
            args.checkpoint_root / dataset / f"seed_{seed}" / f"width_{width}.pt"
        )

        @torch.no_grad()
        def forward():
            return model(image)

        functions = {
            "forward": forward,
            "forward_and_detector": lambda: input_signals(model, image),
        }
        result = {}
        for name, fn in functions.items():
            for _ in range(5):
                fn()
            values = []
            for _ in range(30):
                start = time.perf_counter()
                fn()
                values.append(1000 * (time.perf_counter() - start))
            result[name] = {
                "median_ms": float(np.median(values)),
                "mean_ms": float(np.mean(values)),
                "repeats_ms": values,
                "batch_size": 1,
                "includes_preprocessing": False,
                "own_training_and_attack_jobs_finished": True,
            }
        write_json(folder / "timing.json", result)
        print(dataset, seed, {k: v["median_ms"] for k, v in result.items()}, flush=True)
