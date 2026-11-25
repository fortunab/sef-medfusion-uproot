# SeF MedFusion Uproot

Reproducible histology and cytology classification with gradient–confidence monitoring, RDF provenance, SHACL validation, and a local event monitor. The accompanying paper includes the requested Springer LLNCS variant and a second variant using the official AINA 2027 `svproc` template. Both target a maximum of 12 pages including references.

This is a documented reconstruction of the supplied manuscript. Experimental numbers come from the runs in `results/`. The implementation does not reproduce the separate activation-diversity AD-ZeroNAS search procedure, infer TNM stages from image classes, or establish clinical validity.

## Install

Use Python 3.12. Create and activate a virtual environment, then install CPU PyTorch and the package:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux or macOS: source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
```

`requirements-tested.txt` records the exact direct package versions used for the delivered experiments. Other compatible versions can produce different floating-point results. The reference study uses CPU execution and four PyTorch threads.

## Reproduce the study

Run commands from the repository root:

```bash
python scripts/reproduce.py
```

The command downloads data, prepares partitions, trains all candidates, evaluates attacks, runs the federated and source-family holdout analyses, independently repeats selected-model training, replays attack minibatches, generates tables and figures, and verifies results. Training and attack stages resume from verified completed outputs. Change the results directory when changing the protocol. The individual `scripts/recheck_training.py` and `scripts/recheck_attacks.py` commands can repeat only the replay checks.

The original data archives total approximately 5.44 GB. Extraction needs additional disk space and can take several minutes per solid 7z archive. Data remain in `data/`, which is ignored by Git. Model inputs are 64 × 64 RGB images; these experiments do not use whole-slide images or pretrained models.

Individual stages are available through the CLI:

```bash
sef-medfusion prepare
sef-medfusion train
sef-medfusion robustness
sef-medfusion federated
sef-medfusion report
sef-medfusion verify
```

Use `--dataset kather2016` or `--dataset sipakmed` to restrict data, training, or attack stages. `--data-root`, `--results-root`, and `--checkpoint-root` select storage locations. The report command writes to `paper/generated/` and `paper/figures/` relative to the current repository root. Archive locations and download integrity are recorded in the manifests; the Kather archive is also checked against its provider's MD5.

## Inspect the delivered results

- `results/summary.json`: aggregate results used in the paper.
- `results/verification.json`: independent metric and integrity checks.
- `results/manifests/`: source filenames, classes, source groups, partitions, and hashes.
- `results/<dataset>/seed_<seed>/width_<width>.json`: training history, proxy, checkpoint hash, and clean metrics.
- `*_predictions.npz`: per-image labels, probabilities, and source IDs.
- `*_signals.npz`: attack indices, labels, attack-success masks, input hashes, probabilities, and detector signals.
- `calibration.json`: fixed gradient normalizer and clean-data thresholds.
- `provenance.ttl`, `events.jsonl`, `semantic_validation.json`: graph, matching events, and graph checks for seed 17.
- `reproduction_check.json`: the independent selected-model training repeat.
- `checkpoints/`: all trained reference models and the two federated models.

The numerical arrays contain no redistributed source images. They can be loaded with `numpy.load(path, allow_pickle=False)`. See [the protocol](docs/PROTOCOL.md) for metric definitions and [the reconstruction audit](docs/RECONSTRUCTION_AUDIT.md) for differences from the initial paper.

## Predict and monitor

```bash
sef-medfusion predict --checkpoint checkpoints/kather2016/seed_17/width_8.pt --calibration results/kather2016/seed_17/calibration.json --image example.png --output demo/prediction.json
sef-medfusion monitor --events demo/events.jsonl
```

Open `http://127.0.0.1:8765` while the monitor is running. Each prediction writes JSON, a validated RDF graph, and an event appended to the JSONL file. A flag means that the input exceeds the calibrated review threshold; it does not establish an adversarial cause. TNM status remains `not_available`. The Python API can preserve an externally supplied TNM record with its source, system, and edition.

## Build the paper

Upload the separate Overleaf ZIP, or upload the contents of `paper/`. Select `main_aina.tex` for the official AINA template or `main.tex` for the requested LLNCS variant. Select pdfLaTeX and a current TeX Live version. All class and bibliography-style files are included without changes. Generated tables and vector figures are already present, so Overleaf compilation does not run Python or download datasets.

Local compilation:

```bash
cd paper
latexmk -pdf main_aina.tex
latexmk -pdf main.tex
cd ..
python scripts/check_paper.py paper/main.pdf
python scripts/check_paper.py paper/main_aina.pdf
```

Tectonic is also supported: `tectonic --keep-logs --keep-intermediates paper/main.tex`. The delivered PDF was compiled locally and visually checked. Author, ORCID, affiliation, and funding details were retained from the supplied manuscript; review them before submission. See [the AINA preparation notes](docs/AINA_2027.md).

## Data attribution and scope

Kather 2016: [Zenodo, DOI 10.5281/zenodo.53169](https://zenodo.org/records/53169), CC BY 4.0; cite Kather et al., *Multi-class texture analysis in colorectal cancer histology*, Scientific Reports 6, 27988 (2016).

SIPaKMeD: [the original University of Ioannina release](https://www.cse.uoi.gr/~marina/sipakmed.html); the provider permits experimental use with citation to Plissiti et al., ICIP 2018, pp. 3144–3148, DOI 10.1109/ICIP.2018.8451588. No additional dataset license is inferred or granted here.

The code license is MIT. Manuscript rights belong to its authors. Springer template files and third-party datasets retain their respective terms and notices. No remote repository or conference submission is created by this artifact.

