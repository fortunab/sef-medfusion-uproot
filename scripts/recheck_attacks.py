"""Replay the first attack minibatch for seed 17 and check exact input hashes."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from sef_medfusion.attacks import attack_batches
from sef_medfusion.data import prepare_data, load_data
from sef_medfusion.detector import Calibration, collect_signals
from sef_medfusion.experiment import load_model
from sef_medfusion.utils import seed_all, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=Path("data"))
parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
parser.add_argument("--results-root", type=Path, default=Path("results"))
args = parser.parse_args()
config = json.loads((args.results_root / "config.json").read_text())
report = {"cases": []}
for dataset in config["datasets"]:
    seed = config["seeds"][0]
    seed_all(seed, config["threads"])
    cache = prepare_data(
        dataset, args.data_root, config["image_size"], config["split_seed"]
    )
    x, y, split, _ = load_data(cache)
    test_x, test_y = x[split == "test"], y[split == "test"]
    folder = args.results_root / dataset / f"seed_{seed}"
    width = json.loads((folder / "proxy.json").read_text())["selected_width"]
    model, _ = load_model(
        args.checkpoint_root / dataset / f"seed_{seed}" / f"width_{width}.pt"
    )
    calibration = Calibration(**json.loads((folder / "calibration.json").read_text()))
    for tag in [
        "fgsm_04",
        "fgsm_08",
        "pgd20_04",
        "pgd20_08",
        "gaussian_08",
        "adaptive_08",
    ]:
        record = json.loads((folder / f"{tag}.json").read_text())
        method, epsilon = record["attack"], record["epsilon"]
        with np.load(folder / f"{tag}_signals.npz", allow_pickle=False) as z:
            ids = z["indices"][:32]
            origin, labels = test_x[ids], test_y[ids]
            torch.manual_seed(
                seed
                + round(epsilon * 255)
                + {"fgsm": 100, "pgd20": 200, "gaussian": 300, "adaptive": 400}[method]
            )
            if method == "gaussian":
                replay = (origin + torch.randn_like(origin) * epsilon).clamp(0, 1)
            else:
                replay = attack_batches(
                    model,
                    origin,
                    labels,
                    batch_size=32,
                    epsilon=epsilon,
                    steps=1 if method == "fgsm" else config["pgd_steps"],
                    step_size=(
                        epsilon if method == "fgsm" else config["attack_step_size"]
                    ),
                    random_start=method != "fgsm",
                    calibration=calibration if method == "adaptive" else None,
                    adaptive_lambda=config["adaptive_lambda"],
                )
                assert float((replay - origin).abs().max()) <= epsilon + 1e-6
            hashes = np.array(
                [
                    hashlib.sha256(image.numpy().tobytes()).hexdigest()
                    for image in replay
                ]
            )
            np.testing.assert_array_equal(hashes, z["input_sha256"][:32])
            signal = collect_signals(model, replay)
            difference = float(
                np.abs(signal["probabilities"].numpy() - z["probabilities"][:32]).max()
            )
            assert difference < 2e-6
        item = {
            "dataset": dataset,
            "seed": seed,
            "condition": tag,
            "n": len(ids),
            "input_hashes_equal": True,
            "max_probability_difference": difference,
            "runtime_bound_checked": method != "gaussian",
        }
        report["cases"].append(item)
        print("REPLAYED", item, flush=True)
report["passed"] = True
write_json(args.results_root / "attack_replay_check.json", report)
