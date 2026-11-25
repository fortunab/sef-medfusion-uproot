from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SeF-MedFusion reproducible research pipeline"
    )
    parser.add_argument(
        "command",
        choices=[
            "download",
            "prepare",
            "train",
            "robustness",
            "federated",
            "report",
            "verify",
            "predict",
            "monitor",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--dataset", choices=["kather2016", "sipakmed"], action="append"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("prediction.json"))
    parser.add_argument("--events", type=Path, default=Path("events.jsonl"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from .utils import seed_all

    seed_all(config["seeds"][0], config["threads"])
    if args.command in ["download", "prepare"]:
        from .data import fetch_data, prepare_data

        for dataset in args.dataset or config["datasets"]:
            if args.command == "download":
                fetch_data(dataset, args.data_root)
            else:
                prepare_data(
                    dataset, args.data_root, config["image_size"], config["split_seed"]
                )
    elif args.command in ["train", "robustness", "federated"]:
        from .experiment import train_experiments, robustness_experiments, run_federated

        {
            "train": train_experiments,
            "robustness": robustness_experiments,
            "federated": run_federated,
        }[args.command](
            config,
            args.data_root,
            args.results_root,
            args.checkpoint_root,
            args.dataset,
        )
    elif args.command == "report":
        from .report import generate_report

        generate_report(args.results_root, Path("paper"))
    elif args.command == "verify":
        from .verify import verify_results

        verify_results(args.results_root, args.checkpoint_root)
    elif args.command == "predict":
        if not all([args.checkpoint, args.calibration, args.image]):
            parser.error("predict needs --checkpoint --calibration --image")
        from .inference import predict_image

        predict_image(args.checkpoint, args.calibration, args.image, args.output)
    elif args.command == "monitor":
        from .monitor import serve

        serve(args.events, args.port)


if __name__ == "__main__":
    main()
