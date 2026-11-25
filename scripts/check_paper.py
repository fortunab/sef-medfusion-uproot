"""Enforce the inclusive 12-page limit and check LaTeX build diagnostics."""

import argparse, json, re
from pathlib import Path
from pypdf import PdfReader

parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path, nargs="?", default=Path("paper/main.pdf"))
args = parser.parse_args()
reader = PdfReader(args.pdf)
assert (
    1 <= len(reader.pages) <= 12
), f"{len(reader.pages)} pages; maximum is 12 including references"
text = "\n".join(page.extract_text() or "" for page in reader.pages)
assert "References" in text
assert "??" not in text, "Unresolved reference in PDF"
log = args.pdf.with_suffix(".log")
if log.exists():
    content = log.read_text(encoding="utf-8", errors="replace")
    assert not re.search(
        r"(undefined references|Citation .* undefined|Reference .* undefined|Overfull \\hbox)",
        content,
        re.I,
    ), "Inspect LaTeX diagnostics"
report = {
    "document": args.pdf.name,
    "pages_including_references": len(reader.pages),
    "within_12_page_limit": True,
    "unresolved_references": False,
    "visual_inspection": "see delivery validation report",
}
report_name = (
    "page_check.json" if args.pdf.stem == "main" else args.pdf.stem + "_page_check.json"
)
args.pdf.with_name(report_name).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
