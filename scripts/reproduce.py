"""Run the entire study with fail-fast subprocesses and resumable stages."""

import argparse, subprocess, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default="data")
parser.add_argument("--checkpoint-root", default="checkpoints")
parser.add_argument("--results-root", default="results")
parser.add_argument(
    "--stages",
    nargs="+",
    default=[
        "prepare",
        "train",
        "robustness",
        "federated",
        "source-holdout",
        "recheck-training",
        "recheck-attacks",
        "benchmark",
        "report",
        "verify",
    ],
)
args = parser.parse_args()
for stage in args.stages:
    scripts = {
        "source-holdout": "source_holdout.py",
        "recheck-training": "recheck_training.py",
        "recheck-attacks": "recheck_attacks.py",
        "benchmark": "benchmark.py",
    }
    command = (
        [sys.executable, str(Path("scripts") / scripts[stage])]
        if stage in scripts
        else [sys.executable, "-m", "sef_medfusion.cli", stage]
    )
    subprocess.run(
        command
        + [
            "--data-root",
            args.data_root,
            "--checkpoint-root",
            args.checkpoint_root,
            "--results-root",
            args.results_root,
        ],
        check=True,
    )
