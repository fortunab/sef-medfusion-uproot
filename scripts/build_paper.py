"""Compile both template variants and enforce the inclusive page limit."""

import argparse, shutil, subprocess, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--engine", choices=["auto", "latexmk", "tectonic", "pdflatex"], default="auto"
)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
engine = args.engine
if engine == "auto":
    engine = next(
        (name for name in ["latexmk", "tectonic", "pdflatex"] if shutil.which(name)),
        None,
    )
    if engine is None:
        raise SystemExit(
            "Install TeX Live/latexmk or Tectonic, or compile the Overleaf ZIP."
        )
for filename in ["main.tex", "main_aina.tex"]:
    if engine == "latexmk":
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", filename],
            cwd=root / "paper",
            check=True,
        )
    elif engine == "tectonic":
        subprocess.run(
            ["tectonic", "--keep-logs", "--keep-intermediates", filename],
            cwd=root / "paper",
            check=True,
        )
    else:
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", filename]
        subprocess.run(command, cwd=root / "paper", check=True)
        subprocess.run(["bibtex", Path(filename).stem], cwd=root / "paper", check=True)
        subprocess.run(command, cwd=root / "paper", check=True)
        subprocess.run(command, cwd=root / "paper", check=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "check_paper.py"),
            str(root / "paper" / Path(filename).with_suffix(".pdf")),
        ],
        check=True,
    )
