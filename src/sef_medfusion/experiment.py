from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from .attacks import attack_batches
from .data import prepare_data, load_data
from .detector import calibrate, collect_signals, input_signals, scores
from .metrics import classification, detection
from .models import CompactCNN, parameter_count, proxy_score
from .training import train, predict, federated_simulation
from .utils import seed_all, sha256, write_json, environment


def load_model(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = CompactCNN(bundle["width"], bundle["classes"])
    model.load_state_dict(bundle["state_dict"])
    return model.eval(), bundle


def train_experiments(config, data_root, results_root, checkpoint_root, datasets=None):
    results_root, checkpoint_root = Path(results_root), Path(checkpoint_root)
    results_root.mkdir(parents=True, exist_ok=True)
    config_path = results_root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(
            "Results directory has a different config; use a fresh directory"
        )
    write_json(config_path, config)
    for dataset in datasets or config["datasets"]:
        cache = prepare_data(
            dataset, Path(data_root), config["image_size"], config["split_seed"]
        )
        x, y, split, manifest = load_data(cache)
        write_json(results_root / "manifests" / f"{dataset}.json", manifest)
        tensors = {
            s: (x[split == s], y[split == s])
            for s in ["train", "tune", "calib", "test"]
        }
        for seed in config["seeds"]:
            folder = results_root / dataset / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            seed_all(seed, config["threads"])
            # A fixed training-only probe minibatch sequence shared by candidates.
            order = torch.randperm(len(tensors["train"][0]))
            batches = [
                (tensors["train"][0][idx], tensors["train"][1][idx])
                for idx in order[
                    : config["proxy_batches"] * config["batch_size"]
                ].split(config["batch_size"])
            ]
            proxies = {}
            for width in config["widths"]:
                seed_all(seed, config["threads"])
                model = CompactCNN(width, len(manifest["classes"]))
                proxies[str(width)] = proxy_score(model, batches)
            selected = min(config["widths"], key=lambda w: proxies[str(w)]["score"])
            write_json(
                folder / "proxy.json",
                {
                    "selected_width": selected,
                    "direction": "minimize",
                    "candidates": proxies,
                },
            )
            print(dataset, "seed", seed, "proxy selects width", selected, flush=True)
            for width in config["widths"]:
                checkpoint = (
                    checkpoint_root / dataset / f"seed_{seed}" / f"width_{width}.pt"
                )
                result_path = folder / f"width_{width}.json"
                if result_path.exists() and checkpoint.exists():
                    previous = json.loads(result_path.read_text())
                    if previous["checkpoint_sha256"] != sha256(checkpoint):
                        raise ValueError("Checkpoint changed")
                    print("Resume: verified", dataset, seed, width, flush=True)
                    continue
                seed_all(seed, config["threads"])
                model = CompactCNN(width, len(manifest["classes"]))

                def progress(row):
                    if row["epoch"] == 1 or row["epoch"] % 5 == 0:
                        print(
                            dataset,
                            seed,
                            width,
                            "epoch",
                            row["epoch"],
                            "loss",
                            round(row["train_loss"], 4),
                            "tuneF1",
                            round(row["tune_macro_f1"], 4),
                            flush=True,
                        )

                info = train(
                    model, *tensors["train"], *tensors["tune"], config, progress
                )
                test_p = predict(model, tensors["test"][0], config["batch_size"])
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
                np.savez_compressed(
                    folder / f"width_{width}_predictions.npz",
                    labels=tensors["test"][1].numpy(),
                    probabilities=test_p,
                    ids=np.array(
                        [r["id"] for r in manifest["records"] if r["split"] == "test"]
                    ),
                )
                result = {
                    "dataset": dataset,
                    "seed": seed,
                    "width": width,
                    "selected_by_proxy": width == selected,
                    "parameters": parameter_count(model),
                    "proxy": proxies[str(width)],
                    "test": classification(tensors["test"][1].numpy(), test_p),
                    "checkpoint": (
                        Path("checkpoints")
                        / dataset
                        / f"seed_{seed}"
                        / f"width_{width}.pt"
                    ).as_posix(),
                    "checkpoint_sha256": sha256(checkpoint),
                    **info,
                }
                write_json(result_path, result)
                print("Completed", dataset, seed, width, result["test"], flush=True)
    write_json(results_root / "environment.json", environment())


def _numpy_signal(signal):
    return {k: v.detach().numpy() for k, v in signal.items()}


def robustness_experiments(
    config, data_root, results_root, checkpoint_root, datasets=None, seeds=None
):
    results_root, checkpoint_root = Path(results_root), Path(checkpoint_root)
    if json.loads((results_root / "config.json").read_text()) != config:
        raise ValueError("Attack configuration must match the training protocol")
    for dataset in datasets or config["datasets"]:
        cache = prepare_data(
            dataset, Path(data_root), config["image_size"], config["split_seed"]
        )
        x, y, split, manifest = load_data(cache)
        test_x, test_y = x[split == "test"], y[split == "test"]
        for seed in seeds or config["seeds"]:
            seed_all(seed, config["threads"])
            folder = results_root / dataset / f"seed_{seed}"
            selected = json.loads((folder / "proxy.json").read_text())["selected_width"]
            checkpoint = (
                checkpoint_root / dataset / f"seed_{seed}" / f"width_{selected}.pt"
            )
            model, bundle = load_model(checkpoint)
            calibration = calibrate(
                collect_signals(model, x[split == "calib"]),
                config["alpha"],
                config["calibration_quantile"],
            )
            write_json(folder / "calibration.json", calibration.to_dict())
            clean = collect_signals(model, test_x)
            clean_scores = scores(clean, calibration.gradient_scale, calibration.alpha)
            clean_correct = clean["probabilities"].argmax(1) == test_y
            np.savez_compressed(
                folder / "clean_signals.npz",
                labels=test_y.numpy(),
                **_numpy_signal(clean),
                **{f"score_{k}": v.numpy() for k, v in clean_scores.items()},
            )
            cases = []
            for epsilon in config["attack_epsilons"]:
                cases += [("fgsm", epsilon), ("pgd20", epsilon)]
            cases += [
                ("gaussian", config["noise_sigma"]),
                ("adaptive", max(config["attack_epsilons"])),
            ]
            for attack_name, epsilon in cases:
                tag = f"{attack_name}_{round(epsilon*255):02d}"
                result_path = folder / f"{tag}.json"
                array_path = folder / f"{tag}_signals.npz"
                if result_path.exists() and array_path.exists():
                    print("Resume: robustness", dataset, seed, tag, flush=True)
                    continue
                start = time.perf_counter()
                ids = torch.arange(len(test_y))
                if attack_name == "adaptive":
                    correct_ids = torch.where(clean_correct)[0]
                    gen = torch.Generator().manual_seed(seed + 10000)
                    ids = correct_ids[
                        torch.randperm(len(correct_ids), generator=gen)[
                            : config["adaptive_samples"]
                        ]
                    ]
                    if not len(ids):
                        continue
                origin, labels = test_x[ids], test_y[ids]
                torch.manual_seed(
                    seed
                    + round(epsilon * 255)
                    + {"fgsm": 100, "pgd20": 200, "gaussian": 300, "adaptive": 400}[
                        attack_name
                    ]
                )
                if attack_name == "gaussian":
                    adversarial = (origin + torch.randn_like(origin) * epsilon).clamp(
                        0, 1
                    )
                else:
                    adversarial = attack_batches(
                        model,
                        origin,
                        labels,
                        batch_size=32,
                        epsilon=epsilon,
                        steps=1 if attack_name == "fgsm" else config["pgd_steps"],
                        step_size=(
                            epsilon
                            if attack_name == "fgsm"
                            else config["attack_step_size"]
                        ),
                        random_start=attack_name != "fgsm",
                        calibration=calibration if attack_name == "adaptive" else None,
                        adaptive_lambda=config["adaptive_lambda"],
                    )
                    assert float((adversarial - origin).abs().max()) <= epsilon + 1e-6
                signal = collect_signals(model, adversarial)
                values = scores(signal, calibration.gradient_scale, calibration.alpha)
                success = clean_correct[ids] & (
                    signal["probabilities"].argmax(1) != labels
                )
                mask_clean = {k: v[ids] for k, v in clean_scores.items()}
                detectors = {
                    name: detection(
                        mask_clean[name].numpy(),
                        value.numpy(),
                        calibration.thresholds[name],
                        success.numpy(),
                    )
                    for name, value in values.items()
                }
                result = {
                    "dataset": dataset,
                    "seed": seed,
                    "selected_width": selected,
                    "attack": attack_name,
                    "epsilon": epsilon,
                    "steps": (
                        0
                        if attack_name == "gaussian"
                        else 1 if attack_name == "fgsm" else config["pgd_steps"]
                    ),
                    "n": len(ids),
                    "classification": classification(
                        labels.numpy(), signal["probabilities"].numpy()
                    ),
                    "clean_correct_n": int(clean_correct[ids].sum()),
                    "attack_success_rate": (
                        float(success.sum() / clean_correct[ids].sum())
                        if clean_correct[ids].any()
                        else None
                    ),
                    "detectors": detectors,
                    "max_pixel_delta": float((adversarial - origin).abs().max()),
                    "elapsed_seconds": time.perf_counter() - start,
                    "checkpoint_sha256": sha256(checkpoint),
                }
                np.savez_compressed(
                    array_path,
                    labels=labels.numpy(),
                    indices=ids.numpy(),
                    successful=success.numpy(),
                    input_sha256=np.array(
                        [
                            hashlib.sha256(v.numpy().tobytes()).hexdigest()
                            for v in adversarial
                        ]
                    ),
                    **_numpy_signal(signal),
                    **{f"score_{k}": v.numpy() for k, v in values.items()},
                )
                write_json(result_path, result)
                print(
                    "Attack",
                    dataset,
                    seed,
                    tag,
                    "acc",
                    round(result["classification"]["accuracy"], 4),
                    "TPR",
                    round(detectors["combined"]["tpr_all"], 4),
                    "FPR",
                    round(detectors["combined"]["fpr"], 4),
                    flush=True,
                )
                if (
                    seed == config["seeds"][0]
                    and attack_name == "pgd20"
                    and epsilon == max(config["attack_epsilons"])
                ):
                    from .semantic import export_run

                    export_run(
                        folder,
                        manifest,
                        bundle,
                        sha256(checkpoint),
                        clean,
                        signal,
                        calibration,
                        seed,
                        config,
                        test_x,
                        adversarial,
                    )
            time_path = folder / "timing.json"
            if not time_path.exists():
                times = {}
                for name, fn in [
                    ("forward", lambda: model(test_x[:1])),
                    ("forward_and_detector", lambda: input_signals(model, test_x[:1])),
                ]:
                    for _ in range(5):
                        fn()
                    measurements = []
                    with torch.enable_grad():
                        for _ in range(30):
                            tick = time.perf_counter()
                            fn()
                            measurements.append(1000 * (time.perf_counter() - tick))
                    times[name] = {
                        "median_ms": float(np.median(measurements)),
                        "mean_ms": float(np.mean(measurements)),
                        "repeats_ms": measurements,
                        "batch_size": 1,
                        "includes_preprocessing": False,
                    }
                write_json(time_path, times)


def run_federated(config, data_root, results_root, checkpoint_root, datasets=None):
    for dataset in datasets or config["datasets"]:
        cache = prepare_data(
            dataset, Path(data_root), config["image_size"], config["split_seed"]
        )
        x, y, split, manifest = load_data(cache)
        seed = config["seeds"][0]
        folder = Path(results_root) / dataset / f"seed_{seed}"
        checkpoint = Path(checkpoint_root) / dataset / f"seed_{seed}" / "federated.pt"
        if (folder / "federated.json").exists():
            previous = json.loads((folder / "federated.json").read_text())
            if sha256(checkpoint) != previous["checkpoint_sha256"]:
                raise ValueError("Federated checkpoint changed")
            model, _ = load_model(checkpoint)
            probabilities = predict(model, x[split == "test"])
            recomputed = classification(y[split == "test"].numpy(), probabilities)
            for key, value in recomputed.items():
                assert abs(value - previous["rounds"][-1][key]) < 2e-6
            np.savez_compressed(
                folder / "federated_predictions.npz",
                labels=y[split == "test"].numpy(),
                probabilities=probabilities,
            )
            continue
        width = json.loads((folder / "proxy.json").read_text())["selected_width"]
        start = time.perf_counter()
        model, results = federated_simulation(
            width,
            len(manifest["classes"]),
            x[split == "train"],
            y[split == "train"],
            x[split == "test"],
            y[split == "test"],
            config,
            seed,
        )
        results["elapsed_seconds"] = time.perf_counter() - start
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
        results["checkpoint_sha256"] = sha256(checkpoint)
        write_json(folder / "federated.json", results)
        np.savez_compressed(
            folder / "federated_predictions.npz",
            labels=y[split == "test"].numpy(),
            probabilities=predict(model, x[split == "test"]),
        )
