import copy
import numpy as np
import pytest
import torch
from pyshacl import validate
from rdflib import URIRef, Literal, RDF, XSD
from sef_medfusion.attacks import attack
from sef_medfusion.data import group_splits
from sef_medfusion.detector import input_signals, collect_signals, calibrate, scores
from sef_medfusion.metrics import classification
from sef_medfusion.models import CompactCNN
from sef_medfusion.semantic import (
    MED,
    new_graph,
    add_prediction,
    shapes,
    attach_external_tnm,
)
from sef_medfusion.training import average_states
from sef_medfusion.utils import seed_all


@pytest.fixture
def model():
    seed_all(123, 2)
    return CompactCNN(8, 3).eval()


def test_per_sample_gradient_does_not_depend_on_batch_composition(model):
    images = torch.rand(4, 3, 16, 16)
    batched = collect_signals(model, images, 4)
    singles = collect_signals(model, images, 1)
    torch.testing.assert_close(
        batched["gradient"], singles["gradient"], rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(batched["confidence"], singles["confidence"])


def test_calibration_fixed_scale_and_bounded_scores(model):
    clean = collect_signals(model, torch.rand(16, 3, 16, 16))
    calibration = calibrate(clean)
    changed = {**clean, "gradient": clean["gradient"] * 100}
    values = scores(changed, calibration.gradient_scale)
    assert (values["combined"] >= 0).all() and (values["combined"] <= 1).all()
    assert (values["gradient"] == 1).all()
    assert calibration.gradient_scale == float(clean["gradient"].max())


@pytest.mark.parametrize("adaptive", [False, True])
def test_attack_preserves_inputs_model_and_pixel_budget(model, adaptive):
    images = torch.rand(3, 3, 16, 16)
    original = images.clone()
    labels = torch.tensor([0, 1, 2])
    before = copy.deepcopy(model.state_dict())
    calibration = calibrate(collect_signals(model, images)) if adaptive else None
    result = attack(
        model, images, labels, epsilon=8 / 255, steps=3, calibration=calibration
    )
    torch.testing.assert_close(images, original)
    assert torch.isfinite(result).all() and (result >= 0).all() and (result <= 1).all()
    assert float((result - images).abs().max()) <= 8 / 255 + 1e-6
    for key, value in model.state_dict().items():
        torch.testing.assert_close(before[key], value)


def test_source_groups_never_cross_partitions():
    records = [
        {"label": label, "group": f"{label}/{group}"}
        for label in range(3)
        for group in range(30)
        for _ in range(3)
    ]
    split = group_splits(records)
    np.testing.assert_array_equal(split, group_splits(records))
    assigned = {}
    for r, s in zip(records, split):
        assigned.setdefault(r["group"], set()).add(str(s))
    assert all(len(v) == 1 for v in assigned.values())
    assert set(split) == {"train", "tune", "calib", "test"}


def test_fedavg_uses_sample_counts_and_keeps_integer_buffers():
    a = {"weight": torch.tensor([1.0, 3.0]), "counter": torch.tensor(4)}
    b = {"weight": torch.tensor([5.0, 7.0]), "counter": torch.tensor(8)}
    avg = average_states([a, b], [1, 3])
    torch.testing.assert_close(avg["weight"], torch.tensor([4.0, 6.0]))
    assert avg["counter"].dtype == torch.int64 and avg["counter"] == 8
    torch.testing.assert_close(a["weight"], torch.tensor([1.0, 3.0]))


def _valid_graph():
    graph = new_graph()
    event = {
        "id": "test",
        "condition": "clean",
        "predicted_class": "tissue",
        "confidence": 0.8,
        "gradient_norm": 0.5,
        "score": 0.3,
        "flagged": False,
    }
    node = add_prediction(graph, event, "a" * 64, "test-run", "b" * 64)
    return graph, node


def test_valid_rdf_has_provenance_and_no_invented_stage():
    graph, node = _valid_graph()
    assert validate(graph, shacl_graph=shapes())[0]
    assert str(graph.value(node, MED.stageStatus)) == "not_available"
    assert not list(graph.objects(node, MED.externalTNM))


@pytest.mark.parametrize(
    "mutation",
    [
        "confidence",
        "missing_model",
        "score",
        "missing_sample",
        "missing_stage_evidence",
    ],
)
def test_semantic_constraints_reject_invalid_outputs(mutation):
    graph, node = _valid_graph()
    if mutation == "confidence":
        graph.set((node, MED.confidence, Literal(1.1, datatype=XSD.double)))
    if mutation == "score":
        graph.set((node, MED.adversarialScore, Literal(-0.1, datatype=XSD.double)))
    if mutation == "missing_model":
        graph.remove((node, MED.model, None))
    if mutation == "missing_sample":
        graph.remove((node, MED.sample, None))
    if mutation == "missing_stage_evidence":
        graph.set(
            (node, MED.stageStatus, Literal("externally_supplied", datatype=XSD.string))
        )
    assert not validate(graph, shacl_graph=shapes())[0]


def test_tnm_import_requires_external_evidence_and_preserves_values():
    graph, node = _valid_graph()
    with pytest.raises(ValueError):
        attach_external_tnm(graph, node, {"T": "T1", "N": "N0", "M": "M0"})
    metadata = {
        "T": "T2",
        "N": "N1",
        "M": "M0",
        "system": "external test record",
        "edition": "example",
        "source": "urn:test:source",
    }
    attach_external_tnm(graph, node, metadata)
    assert validate(graph, shacl_graph=shapes())[0]
    stage = graph.value(node, MED.externalTNM)
    assert str(graph.value(stage, MED.T)) == "T2"
    attach_external_tnm(graph, node, {**metadata, "T": "T3"})
    assert str(graph.value(stage, MED.T)) == "T3"
    assert validate(graph, shacl_graph=shapes())[0]
    attach_external_tnm(graph, node, None)
    assert validate(graph, shacl_graph=shapes())[0]
    assert not list(graph.objects(node, MED.externalTNM))


def test_macro_f1_includes_all_dataset_classes():
    metric = classification(
        np.array([0, 0]), np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    assert metric["accuracy"] == 1 and metric["macro_f1"] == pytest.approx(1 / 3)
