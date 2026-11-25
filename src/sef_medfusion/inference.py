from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .detector import Calibration, input_signals, scores
from .experiment import load_model
from .semantic import new_graph, add_prediction, shapes
from .utils import sha256, write_json


def predict_image(checkpoint, calibration_file, image_file, output):
    from pyshacl import validate

    model, bundle = load_model(checkpoint)
    calibration = Calibration(
        **json.loads(Path(calibration_file).read_text(encoding="utf-8"))
    )
    with Image.open(image_file) as image:
        array = np.asarray(
            image.convert("RGB").resize(
                (bundle["image_size"],) * 2, Image.Resampling.BILINEAR
            )
        ).copy()
    x = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255
    signal = input_signals(model, x)
    score = float(
        scores(signal, calibration.gradient_scale, calibration.alpha)[
            "combined"
        ].detach()[0]
    )
    model_hash = sha256(checkpoint)
    event = {
        "id": Path(image_file).name,
        "condition": "external_input",
        "dataset": "external",
        "predicted_class": bundle["class_names"][
            int(signal["probabilities"][0].argmax())
        ],
        "probabilities": dict(
            zip(bundle["class_names"], signal["probabilities"][0].detach().tolist())
        ),
        "confidence": float(signal["confidence"][0].detach()),
        "gradient_norm": float(signal["gradient"][0].detach()),
        "score": score,
        "flagged": score > calibration.thresholds["combined"],
        "stage_status": "not_available",
        "model_sha256": model_hash,
        "source_file_sha256": sha256(image_file),
        "input_sha256": hashlib.sha256(x[0].numpy().tobytes()).hexdigest(),
    }
    graph = new_graph()
    add_prediction(
        graph,
        event,
        model_hash,
        "inference-" + sha256(image_file)[:16],
        event["input_sha256"],
    )
    conforms, _, report = validate(graph, shacl_graph=shapes())
    if not conforms:
        raise ValueError(report)
    write_json(output, event)
    graph.serialize(Path(output).with_suffix(".ttl"), format="turtle")
    events = Path(output).parent / "events.jsonl"
    with events.open("a", encoding="utf-8") as target:
        target.write(json.dumps(event) + "\n")
    print(json.dumps(event, indent=2))
