from __future__ import annotations
import hashlib, json, time
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote, urlparse
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD
from pyshacl import validate
from .detector import scores
from .utils import write_json

# Proposed local vocabulary. No claim that the w3id namespace is registered.
MED = Namespace("https://w3id.org/sef-medfusion/research#")
PROV = Namespace("http://www.w3.org/ns/prov#")


def shapes():
    return Graph().parse(
        data=files("sef_medfusion")
        .joinpath("resources/shapes.ttl")
        .read_text(encoding="utf-8"),
        format="turtle",
    )


def new_graph():
    graph = Graph()
    graph.bind("med", MED)
    graph.bind("prov", PROV)
    graph.parse(
        data=files("sef_medfusion")
        .joinpath("resources/ontology.ttl")
        .read_text(encoding="utf-8"),
        format="turtle",
    )
    return graph


def attach_external_tnm(graph, prediction, metadata=None):
    """Preserve an externally supplied TNM record; never infer one from cells."""
    if metadata is None:
        graph.set(
            (prediction, MED.stageStatus, Literal("not_available", datatype=XSD.string))
        )
        graph.remove((prediction, MED.externalTNM, None))
        return
    required = ["T", "N", "M", "system", "edition", "source"]
    if any(
        not isinstance(metadata.get(k), str) or not metadata[k].strip()
        for k in required
    ):
        raise ValueError("External TNM needs T,N,M,system,edition,source")
    if urlparse(metadata["source"]).scheme not in ["http", "https", "urn"]:
        raise ValueError("External TNM source must be an absolute IRI")
    node = URIRef(str(prediction) + "/external-tnm")
    graph.add((node, RDF.type, MED.ExternalTNM))
    for field, predicate in [
        ("T", MED.T),
        ("N", MED.N),
        ("M", MED.M),
        ("system", MED.stagingSystem),
        ("edition", MED.stagingEdition),
    ]:
        graph.set((node, predicate, Literal(metadata[field], datatype=XSD.string)))
    graph.set((node, PROV.hadPrimarySource, URIRef(metadata["source"])))
    graph.set((prediction, MED.externalTNM, node))
    graph.set(
        (
            prediction,
            MED.stageStatus,
            Literal("externally_supplied", datatype=XSD.string),
        )
    )


def add_prediction(graph, event, model_hash, run_id, file_hash, external_tnm=None):
    sample = URIRef("urn:sha256:" + file_hash)
    model = URIRef("urn:sha256:" + model_hash)
    run = URIRef("urn:sef:run:" + run_id)
    prediction = URIRef(
        "urn:sef:prediction:"
        + hashlib.sha256(
            (run_id + event["id"] + event["condition"]).encode()
        ).hexdigest()
    )
    for node, kind in [
        (sample, MED.Sample),
        (model, MED.Model),
        (run, MED.ProcessingRun),
        (prediction, MED.Prediction),
    ]:
        graph.add((node, RDF.type, kind))
    graph.add((sample, MED.sha256, Literal(file_hash)))
    graph.add((model, MED.sha256, Literal(model_hash)))
    graph.add((prediction, MED.sample, sample))
    graph.add((prediction, MED.model, model))
    graph.add((prediction, PROV.wasDerivedFrom, sample))
    graph.add((prediction, PROV.wasGeneratedBy, run))
    if event.get("source_file_sha256"):
        source = URIRef("urn:sha256:" + event["source_file_sha256"])
        graph.add((source, RDF.type, PROV.Entity))
        graph.add((source, MED.sha256, Literal(event["source_file_sha256"])))
        graph.add((sample, PROV.wasDerivedFrom, source))
    graph.add((run, PROV.used, sample))
    graph.add((run, PROV.used, model))
    for predicate, key in [
        (MED.confidence, "confidence"),
        (MED.gradientNorm, "gradient_norm"),
        (MED.adversarialScore, "score"),
    ]:
        graph.add(
            (prediction, predicate, Literal(float(event[key]), datatype=XSD.double))
        )
    graph.add(
        (
            prediction,
            MED.classLabel,
            Literal(event["predicted_class"], datatype=XSD.string),
        )
    )
    graph.add(
        (prediction, MED.flagged, Literal(bool(event["flagged"]), datatype=XSD.boolean))
    )
    graph.add(
        (
            prediction,
            MED.inputCondition,
            Literal(event["condition"], datatype=XSD.string),
        )
    )
    attach_external_tnm(graph, prediction, external_tnm)
    return prediction


FLAGGED_QUERY = """
PREFIX med: <https://w3id.org/sef-medfusion/research#>
SELECT (COUNT(?prediction) AS ?count) WHERE {
 ?prediction a med:Prediction ; med:flagged true .
}
"""


def export_run(
    folder,
    manifest,
    bundle,
    model_hash,
    clean,
    adversarial,
    calibration,
    seed,
    config,
    clean_images,
    adversarial_images,
):
    folder = Path(folder)
    graph = new_graph()
    records = [r for r in manifest["records"] if r["split"] == "test"]
    events = []
    run_id = f"{manifest['dataset']}-{seed}-{model_hash[:12]}"
    run = URIRef("urn:sef:run:" + run_id)
    graph.add((run, MED.seed, Literal(seed, datatype=XSD.integer)))
    graph.add(
        (
            run,
            MED.configuration,
            Literal(json.dumps(config, sort_keys=True), datatype=XSD.string),
        )
    )
    graph.add(
        (
            run,
            PROV.startedAtTime,
            Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime),
        )
    )
    dataset = URIRef("urn:sef:dataset:" + manifest["dataset"])
    graph.add((dataset, RDF.type, MED.Dataset))
    dataset_source = {
        "kather2016": "https://zenodo.org/records/53169",
        "sipakmed": "https://www.cse.uoi.gr/~marina/sipakmed.html",
    }[manifest["dataset"]]
    graph.add((dataset, PROV.hadPrimarySource, URIRef(dataset_source)))
    started = time.perf_counter()
    for condition, signal, images in [
        ("clean", clean, clean_images),
        ("pgd20_eps8", adversarial, adversarial_images),
    ]:
        combined = scores(signal, calibration.gradient_scale, calibration.alpha)[
            "combined"
        ]
        for i, record in enumerate(records):
            event = {
                "id": record["id"],
                "dataset": manifest["dataset"],
                "seed": seed,
                "condition": condition,
                "predicted_class": bundle["class_names"][
                    int(signal["probabilities"][i].argmax())
                ],
                "confidence": float(signal["confidence"][i]),
                "gradient_norm": float(signal["gradient"][i]),
                "score": float(combined[i]),
                "flagged": bool(combined[i] > calibration.thresholds["combined"]),
                "stage_status": "not_available",
                "model_sha256": model_hash,
                "source_file_sha256": record["file_sha256"],
                "input_sha256": hashlib.sha256(images[i].numpy().tobytes()).hexdigest(),
            }
            prediction = add_prediction(
                graph, event, model_hash, run_id, event["input_sha256"]
            )
            graph.add(
                (URIRef("urn:sha256:" + event["input_sha256"]), MED.dataset, dataset)
            )
            events.append(event)
    insertion = time.perf_counter() - started
    graph.serialize(folder / "provenance.ttl", format="turtle")
    (folder / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    started = time.perf_counter()
    conforms, report_graph, report_text = validate(
        graph, shacl_graph=shapes(), inference="none"
    )
    validation_seconds = time.perf_counter() - started
    if not conforms:
        raise AssertionError(report_text)
    query_count = int(next(iter(graph.query(FLAGGED_QUERY)))[0])
    assert query_count == sum(event["flagged"] for event in events)
    # Negative control on a copy, leaving the exported valid graph unchanged.
    mutated = Graph()
    for triple in graph:
        mutated.add(triple)
    node = next(mutated.subjects(RDF.type, MED.Prediction))
    mutated.set((node, MED.confidence, Literal(1.5, datatype=XSD.double)))
    invalid_conforms, _, _ = validate(mutated, shacl_graph=shapes(), inference="none")
    assert not invalid_conforms
    stats = {
        "predictions": len(events),
        "triples": len(graph),
        "shacl_conforms": bool(conforms),
        "negative_control_rejected": not bool(invalid_conforms),
        "sparql_flag_count": query_count,
        "direct_flag_count": sum(e["flagged"] for e in events),
        "insertion_seconds": insertion,
        "validation_seconds": validation_seconds,
        "stage_values_inferred": 0,
        "model_sha256": model_hash,
    }
    write_json(folder / "semantic_validation.json", stats)
    (folder / "shacl_report.txt").write_text(report_text, encoding="utf-8")
    print("Semantic export", manifest["dataset"], stats, flush=True)
