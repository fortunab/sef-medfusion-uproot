from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
from .utils import write_json

NAMES = {"kather2016": "Kather 2016", "sipakmed": "SIPaKMeD"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_sd(values, percent=False, digits=1):
    data = np.asarray(values, dtype=float) * (100 if percent else 1)
    return f"{data.mean():.{digits}f} $\\pm$ {data.std(ddof=1):.{digits}f}"


def latex_table(path, caption, label, columns, headers, rows):
    text = (
        "\\begin{table}[t]\n\\centering\n\\caption{"
        + caption
        + "}\n\\label{"
        + label
        + "}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
    )
    text += (
        "\\begin{tabular}{"
        + columns
        + "}\n\\toprule\n"
        + " & ".join(headers)
        + " \\\\\n\\midrule\n"
    )
    text += "\n".join(" & ".join(str(v) for v in row) + " \\\\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    Path(path).write_text(text, encoding="utf-8")


def generate_report(results_root: Path, paper_root: Path):
    results_root, paper_root = Path(results_root), Path(paper_root)
    config = read(results_root / "config.json")
    generated = paper_root / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    figures = paper_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {"datasets": {}, "seeds": config["seeds"]}
    class_rows = []
    attack_rows = []
    detector_rows = []
    system_rows = []
    for dataset in config["datasets"]:
        manifest = read(results_root / "manifests" / f"{dataset}.json")
        counts = manifest["split_counts"]
        rows.append(
            [
                NAMES[dataset],
                len(manifest["classes"]),
                *[counts[k] for k in ["train", "tune", "calib", "test"]],
            ]
        )
        selected = [
            read(results_root / dataset / f"seed_{s}" / "proxy.json")["selected_width"]
            for s in config["seeds"]
        ]
        runs = [
            read(results_root / dataset / f"seed_{s}" / f"width_{w}.json")
            for s, w in zip(config["seeds"], selected)
        ]
        data = {
            "selected_widths": selected,
            "selected_test": {
                key: {
                    "mean": float(np.mean([r["test"][key] for r in runs])),
                    "sd": float(np.std([r["test"][key] for r in runs], ddof=1)),
                }
                for key in ["accuracy", "macro_f1", "confidence", "ece_15"]
            },
        }
        summary["datasets"][dataset] = data
        for width in config["widths"]:
            candidate = [
                read(results_root / dataset / f"seed_{s}" / f"width_{width}.json")
                for s in config["seeds"]
            ]
            marker = "*" if all(w == width for w in selected) else ""
            class_rows.append(
                [
                    NAMES[dataset],
                    f"{width}{marker}",
                    f"{candidate[0]['parameters']/1000:.1f}",
                    mean_sd([r["test"]["accuracy"] for r in candidate], True),
                    mean_sd([r["test"]["macro_f1"] for r in candidate], True),
                    mean_sd([r["test"]["ece_15"] for r in candidate], True),
                ]
            )
        data["attacks"] = {}
        for tag, title in [
            ("fgsm_04", "FGSM 4"),
            ("fgsm_08", "FGSM 8"),
            ("pgd20_04", "PGD 4"),
            ("pgd20_08", "PGD 8"),
            ("gaussian_08", "Noise 8"),
            ("adaptive_08", "Adaptive 8"),
        ]:
            attack = [
                read(results_root / dataset / f"seed_{s}" / f"{tag}.json")
                for s in config["seeds"]
            ]
            successful = [r["detectors"]["combined"]["tpr_successful"] for r in attack]
            data["attacks"][tag] = {
                "accuracy": float(
                    np.mean([r["classification"]["accuracy"] for r in attack])
                ),
                "tpr_successful": float(np.mean(successful)),
                "fpr": float(
                    np.mean([r["detectors"]["combined"]["fpr"] for r in attack])
                ),
                "auroc": float(
                    np.mean([r["detectors"]["combined"]["auroc"] for r in attack])
                ),
            }
            attack_rows.append(
                [
                    NAMES[dataset],
                    title,
                    mean_sd([r["classification"]["accuracy"] for r in attack], True, 2),
                    mean_sd([r["attack_success_rate"] for r in attack], True, 2),
                    mean_sd(successful, True, 2),
                ]
            )
        for name, title in [
            ("uncertainty", "1 - confidence"),
            ("entropy", "Entropy"),
            ("gradient", "Gradient"),
            ("combined", "Combined"),
        ]:
            values = [
                read(results_root / dataset / f"seed_{s}" / "pgd20_08.json")[
                    "detectors"
                ][name]
                for s in config["seeds"]
            ]
            detector_rows.append(
                [
                    NAMES[dataset],
                    title,
                    mean_sd([r["auroc"] for r in values], False, 3),
                    mean_sd([r["tpr_successful"] for r in values], True, 2),
                    mean_sd([r["fpr"] for r in values], True, 2),
                ]
            )
        semantic = read(
            results_root
            / dataset
            / f"seed_{config['seeds'][0]}"
            / "semantic_validation.json"
        )
        fed = read(
            results_root / dataset / f"seed_{config['seeds'][0]}" / "federated.json"
        )
        times = [
            read(results_root / dataset / f"seed_{s}" / "timing.json")
            for s in config["seeds"]
        ]
        data["semantic"] = semantic
        data["federated"] = fed
        system_rows.append(
            [
                NAMES[dataset],
                semantic["predictions"],
                semantic["triples"],
                f"{1000*semantic['insertion_seconds']/semantic['predictions']:.2f}",
                f"{semantic['validation_seconds']:.2f}",
                f"{np.mean([t['forward_and_detector']['median_ms'] for t in times]):.2f}",
            ]
        )
    latex_table(
        generated / "data_table.tex",
        "Dataset partitions after duplicate removal. SIPaKMeD is split by source cluster; Kather is split at tile level.",
        "tab:data",
        "lrrrrr",
        ["Dataset", "Classes", "Train", "Tune", "Calib.", "Test"],
        rows,
    )
    latex_table(
        generated / "classification_table.tex",
        "Clean test performance (percent, mean $\\pm$ sample SD over three initialization seeds). Width denotes the first block channel count; * marks proxy selection. ECE uses 15 bins.",
        "tab:classification",
        "llrrrr",
        ["Dataset", "Width", "Params (k)", "Accuracy", "Macro F1", "ECE"],
        class_rows,
    )
    latex_table(
        generated / "attack_table.tex",
        "Selected-model stress tests (percent, mean $\\pm$ SD). Budgets are in units of $1/255$. ASR counts new errors among initially correct images; TPR$_s$ is detection on those successful perturbations. Gaussian noise uses the indicated standard deviation. Adaptive uses 128 initially correct test images per seed.",
        "tab:attacks",
        "llrrr",
        ["Dataset", "Condition", "Accuracy", "ASR", "TPR$_s$"],
        attack_rows,
    )
    latex_table(
        generated / "detector_table.tex",
        "Detector ablation on PGD-20 at $8/255$. Every threshold is independently calibrated on clean calibration data at the 95th percentile. AUROC uses all clean and perturbed test images; TPR$_s$ uses successful attacks.",
        "tab:detectors",
        "llrrr",
        ["Dataset", "Detector", "AUROC", "TPR$_s$ (\\%)", "FPR (\\%)"],
        detector_rows,
    )
    latex_table(
        generated / "system_table.tex",
        "Measured system checks for seed 17. Insert is graph construction time per prediction, SHACL is full-graph validation time, and monitor is median forward-plus-detector latency averaged across three seeds (batch size 1).",
        "tab:systems",
        "lrrrrr",
        ["Dataset", "Pred.", "Triples", "Insert (ms)", "SHACL (s)", "Monitor (ms)"],
        system_rows,
    )
    interpretation = []
    below_chance = []
    wins = []
    for dataset in config["datasets"]:
        ablations = {
            name: float(
                np.mean(
                    [
                        read(results_root / dataset / f"seed_{s}" / "pgd20_08.json")[
                            "detectors"
                        ][name]["auroc"]
                        for s in config["seeds"]
                    ]
                )
            )
            for name in ["uncertainty", "entropy", "gradient", "combined"]
        }
        best = max(["uncertainty", "entropy", "gradient"], key=ablations.get)
        wins.append(ablations["combined"] > ablations[best])
        if ablations["combined"] < 0.5:
            below_chance.append(NAMES[dataset])
        interpretation.append(
            f"On {NAMES[dataset]}, mean PGD-$8/255$ AUROC is {ablations['combined']:.3f} for the combined score and {ablations[best]:.3f} for the strongest single-signal baseline ({best})."
        )
    if len(below_chance) == len(config["datasets"]):
        interpretation.append(
            "In each dataset, the mean AUROC of the combined score is below 0.5. Its fixed score direction ranks attacked inputs poorly relative to clean images."
        )
    elif below_chance:
        interpretation.append(
            "The combined-score mean AUROC is below 0.5 for "
            + " and ".join(below_chance)
            + "; its predeclared score direction ranks these attacked inputs poorly relative to clean images."
        )
    if not all(wins):
        interpretation.append(
            "The combination does not provide a uniform AUROC improvement over the strongest single-signal baseline across these datasets. Its contribution must therefore be judged from the measured operating points rather than presumed from combining signals."
        )
    else:
        interpretation.append(
            "The combined score has higher mean AUROC than the tested single-signal baselines in these runs. The small number of seeds and restricted attacks do not establish a general superiority claim."
        )
    interpretation.append(
        "The adaptive condition uses a different, initially correct subset; its classification accuracy and detection rate should not be compared directly with whole-test-set PGD as if their denominators were identical."
    )
    (generated / "interpretation.tex").write_text(
        "\n\n".join(interpretation) + "\n", encoding="utf-8"
    )
    macros = []
    for dataset, prefix in [("kather2016", "K"), ("sipakmed", "S")]:
        d = summary["datasets"][dataset]
        for key, suffix in [("accuracy", "Acc"), ("macro_f1", "Fone")]:
            v = d["selected_test"][key]
            macros.append(
                f"\\newcommand{{\\{prefix}{suffix}}}{{{100*v['mean']:.1f}\\%}}"
            )
        macros.append(
            f"\\newcommand{{\\{prefix}PGDAcc}}{{{100*d['attacks']['pgd20_08']['accuracy']:.2f}\\%}}"
        )
        macros.append(
            f"\\newcommand{{\\{prefix}PGDTPR}}{{{100*d['attacks']['pgd20_08']['tpr_successful']:.2f}\\%}}"
        )
        macros.append(
            f"\\newcommand{{\\{prefix}PGDFPR}}{{{100*d['attacks']['pgd20_08']['fpr']:.2f}\\%}}"
        )
        macros.append(
            f"\\newcommand{{\\{prefix}FedAcc}}{{{100*d['federated']['rounds'][-1]['accuracy']:.1f}\\%}}"
        )
    holdout = read(results_root / "kather_source_holdout" / "results.json")
    summary["source_family_holdout"] = holdout
    macros += [
        f"\\newcommand{{\\KSourceAcc}}{{{100*holdout['classification']['accuracy']:.1f}\\%}}",
        f"\\newcommand{{\\KSourcePGDAcc}}{{{100*holdout['pgd20_08']['accuracy']:.2f}\\%}}",
        f"\\newcommand{{\\KSourceFPR}}{{{100*holdout['combined_detector']['fpr']:.1f}\\%}}",
    ]
    (generated / "numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    write_json(results_root / "summary.json", summary)
    _figures(config, results_root, figures)
    print("Generated LaTeX tables, figures and summary from recorded runs", flush=True)


def _figures(config, root, figures):
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.45))
    colors = {
        "uncertainty": "#a8541d",
        "entropy": "#a05ca6",
        "gradient": "#24846b",
        "combined": "#224d8a",
    }
    for ax, dataset in zip(axes, config["datasets"]):
        folder = root / dataset / f"seed_{config['seeds'][0]}"
        with np.load(
            folder / "clean_signals.npz", allow_pickle=False
        ) as clean, np.load(
            folder / "pgd20_08_signals.npz", allow_pickle=False
        ) as attack:
            for name in ["uncertainty", "entropy", "gradient", "combined"]:
                x = np.r_[clean["score_" + name], attack["score_" + name]]
                labels = np.r_[
                    np.zeros(len(clean["labels"])), np.ones(len(attack["labels"]))
                ]
                fpr, tpr, _ = roc_curve(labels, x)
                ax.plot(
                    fpr,
                    tpr,
                    label=f"{name} ({roc_auc_score(labels,x):.2f})",
                    color=colors[name],
                    lw=1.2,
                )
        ax.plot([0, 1], [0, 1], ":", color="#777777", lw=0.8)
        ax.set(
            title=NAMES[dataset],
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.15)
    fig.tight_layout(pad=0.6)
    fig.savefig(figures / "roc.pdf", bbox_inches="tight")
    fig.savefig(figures / "roc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.2))
    for ax, dataset in zip(axes, config["datasets"]):
        folder = root / dataset / f"seed_{config['seeds'][0]}"
        for filename, label, color, marker in [
            ("clean_signals.npz", "Clean", "#224d8a", "o"),
            ("pgd20_08_signals.npz", "PGD 8/255", "#b54b2c", "x"),
        ]:
            with np.load(folder / filename, allow_pickle=False) as z:
                ax.scatter(
                    z["score_gradient"],
                    z["confidence"],
                    s=5,
                    alpha=0.24,
                    color=color,
                    marker=marker,
                    label=label,
                    rasterized=True,
                )
        calibration = read(folder / "calibration.json")
        xx = np.linspace(0, 1, 100)
        yy = (
            1
            + calibration["alpha"] / (1 - calibration["alpha"]) * xx
            - calibration["thresholds"]["combined"] / (1 - calibration["alpha"])
        )
        ax.plot(xx, yy, "--", color="#333333", lw=0.8, label="Flag boundary")
        ax.set(
            title=NAMES[dataset],
            xlabel="Calibrated gradient (clipped)",
            ylabel="Maximum confidence",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.15)
    fig.tight_layout(pad=0.6)
    fig.savefig(figures / "signals.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3))
    for ax, dataset in zip(axes, config["datasets"]):
        for width in config["widths"]:
            runs = [
                read(root / dataset / f"seed_{s}" / f"width_{width}.json")
                for s in config["seeds"]
            ]
            values = (
                np.array([[r["tune_macro_f1"] for r in run["history"]] for run in runs])
                * 100
            )
            epoch = np.arange(1, values.shape[1] + 1)
            ax.plot(epoch, values.mean(0), label=f"width {width}")
            ax.fill_between(
                epoch,
                values.mean(0) - values.std(0, ddof=1),
                values.mean(0) + values.std(0, ddof=1),
                alpha=0.13,
            )
        ax.set(title=NAMES[dataset], xlabel="Epoch", ylabel="Tuning macro F1 (%)")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=6)
    fig.tight_layout(pad=0.6)
    fig.savefig(figures / "learning.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.9, 1.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    boxes = [
        (0.1, 0.9, 1.7, 1.1, "Public images\nfixed splits\nsource hashes"),
        (2.2, 0.9, 2, 1.1, "Compact CNN pool\ntrain-only proxy\nclassification"),
        (4.7, 0.9, 2, 1.1, "Gradient +\nconfidence score\nreview flags"),
        (7.3, 0.9, 2.5, 1.1, "RDF / PROV-O\nSHACL validation\nevents + monitor"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.08",
                facecolor="#edf2f8",
                edgecolor="#224d8a",
                linewidth=0.8,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.5)
    for start, end in [(1.9, 2.1), (4.3, 4.6), (6.8, 7.2)]:
        ax.annotate(
            "",
            xy=(end, 1.45),
            xytext=(start, 1.45),
            arrowprops={"arrowstyle": "->", "lw": 0.9},
        )
    ax.text(
        5,
        0.2,
        "External TNM records retain their source.\nImage classes do not determine a TNM stage.",
        ha="center",
        fontsize=7.5,
        color="#4b5563",
    )
    fig.tight_layout(pad=0.1)
    fig.savefig(figures / "architecture.pdf", bbox_inches="tight")
    plt.close(fig)
