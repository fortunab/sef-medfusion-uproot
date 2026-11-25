from __future__ import annotations
import hashlib, io, json, re, urllib.request, zipfile
from importlib.resources import files
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from .utils import sha256, write_json

KATHER_URL = "https://zenodo.org/api/records/53169/files/Kather_texture_2016_image_tiles_5000.zip/content"
KATHER_MD5 = "0ddbebfc56344752028fda72602aaade"
SIPAK_CLASSES = [
    "Superficial-Intermediate",
    "Parabasal",
    "Koilocytotic",
    "Metaplastic",
    "Dyskeratotic",
]


def download(url: str, path: Path, md5: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(
            url, headers={"User-Agent": "SeF-MedFusion/0.1"}
        )
        print("Downloading", url, flush=True)
        with urllib.request.urlopen(request, timeout=180) as source, path.with_suffix(
            path.suffix + ".part"
        ).open("wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        path.with_suffix(path.suffix + ".part").replace(path)
    if md5 and hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def fetch_data(dataset: str, root: Path):
    root = Path(root)
    if dataset == "kather2016":
        return [
            download(
                KATHER_URL,
                root / "downloads" / "Kather_texture_2016_image_tiles_5000.zip",
                KATHER_MD5,
            )
        ]
    if dataset == "sipakmed":
        archives = [
            download(
                f"https://www.cse.uoi.gr/~marina/SIPAKMED/im_{name}.7z",
                root / "downloads" / f"im_{name}.7z",
            )
            for name in SIPAK_CLASSES
        ]
        expected = json.loads(
            files("sef_medfusion")
            .joinpath("resources/data_checksums.json")
            .read_text(encoding="utf-8")
        )
        for archive in archives:
            if sha256(archive) != expected[archive.name]:
                raise ValueError(f"Source archive version differs: {archive}")
        return archives
    raise ValueError(dataset)


def group_splits(records: list[dict], seed: int = 2027) -> np.ndarray:
    """60/10/10/20 by source group within each class; no group is reused."""
    rng = np.random.default_rng(seed)
    splits = np.empty(len(records), dtype="<U5")
    for label in sorted({r["label"] for r in records}):
        groups = sorted({r["group"] for r in records if r["label"] == label})
        groups = rng.permutation(groups)
        n = len(groups)
        a, b, c = round(0.6 * n), round(0.7 * n), round(0.8 * n)
        if min(a, b - a, c - b, n - c) <= 0:
            raise ValueError("Too few source groups to split safely")
        mapping = {
            group: split
            for subset, split in zip(
                [groups[:a], groups[a:b], groups[b:c], groups[c:]],
                ["train", "tune", "calib", "test"],
            )
            for group in subset
        }
        for i, r in enumerate(records):
            if r["label"] == label:
                splits[i] = mapping[r["group"]]
    return splits


def _safe_members(names):
    for name in names:
        p = PurePosixPath(name.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
            raise ValueError(f"Unsafe archive member: {name}")


def prepare_data(dataset: str, root: Path, size: int = 64, split_seed: int = 2027):
    root = Path(root)
    cache = root / "processed" / f"{dataset}_{size}_{split_seed}.npz"
    manifest_path = cache.with_suffix(".manifest.json")
    if cache.exists() and manifest_path.exists():
        return cache
    archives = fetch_data(dataset, root)
    items = []
    if dataset == "kather2016":
        with zipfile.ZipFile(archives[0]) as z:
            names = sorted(
                n
                for n in z.namelist()
                if n.lower().endswith((".tif", ".tiff", ".png")) and "__MACOSX" not in n
            )
            _safe_members(names)
            classes = sorted({PurePosixPath(n).parent.name for n in names})
            for name in names:
                raw = z.read(name)
                label = classes.index(PurePosixPath(name).parent.name)
                items.append((name, label, name, raw))
    else:
        import py7zr

        classes = SIPAK_CLASSES
        for label, archive in enumerate(archives):
            dest = root / "raw" / "sipakmed" / classes[label]
            marker = dest / ".extracted"
            if not marker.exists():
                with py7zr.SevenZipFile(archive) as z:
                    names = z.getnames()
                    _safe_members(names)
                    targets = [
                        n
                        for n in names
                        if "cropped" in n.lower() and n.lower().endswith(".bmp")
                    ]
                    if not targets:
                        raise ValueError(f"No cropped cells found in {archive}")
                    z.extract(path=dest, targets=targets)
                marker.write_text(sha256(archive), encoding="utf-8")
            for path in sorted(dest.rglob("*.bmp")):
                if "cropped" not in path.as_posix().lower():
                    continue
                # Official names are cluster_cell.bmp; group by source cluster.
                match = re.match(r"^(\d+)[_\-]", path.stem)
                if not match:
                    raise ValueError(f"Cannot infer source cluster from {path.name}")
                group = classes[label] + "/" + match.group(1)
                name = classes[label] + "/" + path.relative_to(dest).as_posix()
                items.append((name, label, group, path.read_bytes()))
    images, records, seen, duplicates = [], [], {}, []
    for name, label, group, raw in items:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        pixel_hash = hashlib.sha256(
            np.asarray(image).tobytes() + str(image.size).encode()
        ).hexdigest()
        if pixel_hash in seen:
            duplicates.append({"removed": name, "kept": seen[pixel_hash]})
            continue
        seen[pixel_hash] = name
        images.append(np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)))
        records.append(
            {
                "id": name,
                "label": int(label),
                "class": classes[label],
                "group": group,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "pixels_sha256": pixel_hash,
            }
        )
    split = group_splits(records, split_seed)
    for row, value in zip(records, split):
        row["split"] = str(value)
    for a in ["train", "tune", "calib", "test"]:
        groups = {r["group"] for r in records if r["split"] == a}
        for b in ["train", "tune", "calib", "test"]:
            if a != b:
                assert groups.isdisjoint(
                    {r["group"] for r in records if r["split"] == b}
                )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        images=np.stack(images),
        labels=np.array([r["label"] for r in records]),
        split=split,
    )
    manifest = {
        "dataset": dataset,
        "image_size": size,
        "split_seed": split_seed,
        "classes": classes,
        "archives": [
            {"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in archives
        ],
        "split_counts": {
            s: int((split == s).sum()) for s in ["train", "tune", "calib", "test"]
        },
        "duplicates_removed": duplicates,
        "records": records,
    }
    write_json(manifest_path, manifest)
    print(
        "Prepared",
        dataset,
        manifest["split_counts"],
        "duplicates",
        len(duplicates),
        flush=True,
    )
    return cache


def load_data(cache: Path):
    import torch

    with np.load(cache, allow_pickle=False) as z:
        x = (
            torch.from_numpy(z["images"].copy())
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            / 255
        )
        y = torch.from_numpy(z["labels"].copy()).long()
        split = z["split"].copy()
    manifest = json.loads(
        Path(cache).with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    return x, y, split, manifest
