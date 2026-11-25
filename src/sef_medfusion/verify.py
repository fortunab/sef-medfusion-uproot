from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from pyshacl import validate
from rdflib import Graph
from .detector import Calibration, scores
from .experiment import load_model
from .metrics import classification, detection
from .models import parameter_count
from .semantic import FLAGGED_QUERY, shapes
from .utils import sha256, write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _same(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert abs(float(a) - float(b)) < 2e-6, (a, b)


def verify_results(root, checkpoint_root):
    root, checkpoint_root = Path(root), Path(checkpoint_root)
    config = _read(root / "config.json")
    checked = {"classification_runs": 0, "attack_runs": 0, "graphs": 0, "checks": []}
    for dataset in config["datasets"]:
        manifest = _read(root / "manifests" / f"{dataset}.json")
        records = manifest["records"]
        assert len({r["pixels_sha256"] for r in records}) == len(records)
        for split in ["train", "tune", "calib", "test"]:
            groups = {r["group"] for r in records if r["split"] == split}
            other = {r["group"] for r in records if r["split"] != split}
            assert groups.isdisjoint(other)
            assert (
                sum(r["split"] == split for r in records)
                == manifest["split_counts"][split]
            )
        test_ids = [r["id"] for r in records if r["split"] == "test"]
        for seed in config["seeds"]:
            folder = root / dataset / f"seed_{seed}"
            proxy = _read(folder / "proxy.json")
            selected = min(
                config["widths"], key=lambda w: proxy["candidates"][str(w)]["score"]
            )
            assert selected == proxy["selected_width"]
            for width in config["widths"]:
                path = checkpoint_root / dataset / f"seed_{seed}" / f"width_{width}.pt"
                run = _read(folder / f"width_{width}.json")
                assert sha256(path) == run["checkpoint_sha256"]
                model, bundle = load_model(path)
                assert parameter_count(model) == run["parameters"]
                with np.load(
                    folder / f"width_{width}_predictions.npz", allow_pickle=False
                ) as z:
                    assert list(z["ids"]) == test_ids
                    assert np.isfinite(z["probabilities"]).all()
                    np.testing.assert_allclose(z["probabilities"].sum(1), 1, atol=2e-6)
                    assert (
                        z["probabilities"].min() >= 0 and z["probabilities"].max() <= 1
                    )
                    recomputed = classification(z["labels"], z["probabilities"])
                    for key in recomputed:
                        _same(recomputed[key], run["test"][key])
                p = proxy["candidates"][str(width)]
                _same(p["score"], p["parameters"] / p["mean_gradient_norm"])
                checked["classification_runs"] += 1
            calibration = Calibration(**_read(folder / "calibration.json"))
            with np.load(folder / "clean_signals.npz", allow_pickle=False) as clean:
                clean_correct = clean["probabilities"].argmax(1) == clean["labels"]
                expected_gradient = np.minimum(
                    clean["gradient"] / calibration.gradient_scale, 1
                )
                np.testing.assert_allclose(
                    clean["score_gradient"], expected_gradient, atol=2e-6
                )
                np.testing.assert_allclose(
                    clean["score_uncertainty"],
                    1 - clean["probabilities"].max(1),
                    atol=2e-6,
                )
                np.testing.assert_allclose(
                    clean["score_combined"],
                    calibration.alpha * expected_gradient
                    + (1 - calibration.alpha) * clean["score_uncertainty"],
                    atol=2e-6,
                )
                for tag in [
                    "fgsm_04",
                    "fgsm_08",
                    "pgd20_04",
                    "pgd20_08",
                    "gaussian_08",
                    "adaptive_08",
                ]:
                    run = _read(folder / f"{tag}.json")
                    with np.load(
                        folder / f"{tag}_signals.npz", allow_pickle=False
                    ) as z:
                        for key, value in classification(
                            z["labels"], z["probabilities"]
                        ).items():
                            _same(value, run["classification"][key])
                        ids = z["indices"]
                        success = clean_correct[ids] & (
                            z["probabilities"].argmax(1) != z["labels"]
                        )
                        np.testing.assert_array_equal(success, z["successful"])
                        if run["attack"] != "gaussian":
                            assert run["max_pixel_delta"] <= run["epsilon"] + 1e-6
                        for name in ["combined", "uncertainty", "gradient", "entropy"]:
                            values = detection(
                                clean["score_" + name][ids],
                                z["score_" + name],
                                calibration.thresholds[name],
                                success,
                            )
                            for key, value in values.items():
                                _same(value, run["detectors"][name][key])
                        assert len(z["input_sha256"]) == run["n"]
                        expected_gradient = np.minimum(
                            z["gradient"] / calibration.gradient_scale, 1
                        )
                        np.testing.assert_allclose(
                            z["score_gradient"], expected_gradient, atol=2e-6
                        )
                        np.testing.assert_allclose(
                            z["score_uncertainty"],
                            1 - z["probabilities"].max(1),
                            atol=2e-6,
                        )
                        np.testing.assert_allclose(
                            z["score_combined"],
                            calibration.alpha * expected_gradient
                            + (1 - calibration.alpha) * z["score_uncertainty"],
                            atol=2e-6,
                        )
                    checked["attack_runs"] += 1
        folder = root / dataset / f"seed_{config['seeds'][0]}"
        graph = Graph().parse(folder / "provenance.ttl", format="turtle")
        assert validate(graph, shacl_graph=shapes(), inference="none")[0]
        semantic = _read(folder / "semantic_validation.json")
        assert len(graph) == semantic["triples"]
        assert (
            int(next(iter(graph.query(FLAGGED_QUERY)))[0])
            == semantic["direct_flag_count"]
        )
        assert (
            semantic["negative_control_rejected"]
            and semantic["stage_values_inferred"] == 0
        )
        events = [
            json.loads(line)
            for line in (folder / "events.jsonl").read_text().splitlines()
        ]
        assert len(events) == semantic["predictions"]
        assert sum(e["flagged"] for e in events) == semantic["direct_flag_count"]
        fed = _read(folder / "federated.json")
        assert fed["checkpoint_sha256"] == sha256(
            checkpoint_root / dataset / f"seed_{config['seeds'][0]}" / "federated.pt"
        )
        assert sum(fed["client_sizes"]) == manifest["split_counts"]["train"]
        assert len(fed["rounds"]) == config["federated_rounds"]
        with np.load(folder / "federated_predictions.npz", allow_pickle=False) as z:
            for key, value in classification(z["labels"], z["probabilities"]).items():
                _same(value, fed["rounds"][-1][key])
        checked.setdefault("federated_models_checked", 0)
        checked["federated_models_checked"] += 1
        repeat = _read(folder / "reproduction_check.json")
        assert (
            repeat["bitwise_equal"]
            and repeat["max_weight_difference"] == 0
            and repeat["max_probability_difference"] == 0
        )
        checked.setdefault("independent_training_repeats", 0)
        checked["independent_training_repeats"] += 1
        checked["graphs"] += 1
    holdout_folder = root / "kather_source_holdout"
    holdout = _read(holdout_folder / "results.json")
    assert holdout["max_pixel_delta"] <= 8 / 255 + 1e-6
    assert holdout["checkpoint_sha256"] == sha256(
        checkpoint_root / "kather_source_holdout" / "width_8.pt"
    )
    manifest = _read(holdout_folder / "manifest.json")
    for record in manifest["records"]:
        assert manifest["assignments"][record["source_family"]] == record["split"]
    with np.load(holdout_folder / "predictions.npz", allow_pickle=False) as z:
        for key, value in classification(z["labels"], z["probabilities"]).items():
            _same(value, holdout["classification"][key])
        for key, value in classification(
            z["labels"], z["adversarial_probabilities"]
        ).items():
            _same(value, holdout["pgd20_08"][key])
    replay = _read(root / "attack_replay_check.json")
    assert (
        replay["passed"]
        and len(replay["cases"]) == 12
        and all(c["input_hashes_equal"] for c in replay["cases"])
    )
    checked["replayed_attack_minibatches"] = len(replay["cases"])
    checked["source_family_holdout_checked"] = True
    checked["checks"] = [
        "source-group separation",
        "pixel duplicate exclusion",
        "checkpoint SHA-256",
        "probability validity",
        "classification metric recomputation",
        "proxy arithmetic and selection",
        "attack pixel bounds",
        "attack-success labels",
        "all detection metric recomputation",
        "RDF round-trip",
        "SHACL validation",
        "negative controls",
        "SPARQL/event count equality",
        "no TNM stages inferred",
    ]
    checked["passed"] = True
    write_json(root / "verification.json", checked)
    print(json.dumps(checked, indent=2))
