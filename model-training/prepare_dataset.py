"""Prepare NEU-DET for YOLO training.

- Merge IMAGES/ANNOTATIONS with the 30 validation samples from the mirror.
- Parse VOC-style XML into YOLO normalized txt labels.
- Stratified split (by dominant defect class) with a fixed seed: train/val/test.
- Verify class distribution and image-level data leakage (no image in two splits).
- Write provenance manifest with source URL, license note and checksums.
- Dataset files are NOT committed to git (see .gitignore).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

MIRROR_SOURCE = "https://github.com/DISHENGRZH/NEU-DET-Steel-Surface-Defect-Detection"
ORIGINAL_SOURCE = "http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html"
LICENSE_NOTE = (
    "Research dataset by K. Song and Y. Yan (Northeastern University). "
    "Academic use with citation: K. Song and Y. Yan, Applied Surface Science, vol.285, pp.858-864, 2013. "
    "No redistribution restrictions were stated for research use on the official page."
)
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_xml(xml_path: Path) -> list[dict]:
    root = ElementTree.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_INDEX:
            raise ValueError(f"unknown class {name!r} in {xml_path.name}")
        bb = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin", "0")))
        ymin = int(float(bb.findtext("ymin", "0")))
        xmax = int(float(bb.findtext("xmax", "0")))
        ymax = int(float(bb.findtext("ymax", "0")))
        boxes.append({"class": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    if not boxes:
        raise ValueError(f"no object found in {xml_path.name}")
    return boxes


def to_yolo(ann: dict, width: int, height: int) -> str:
    cx = (ann["xmin"] + ann["xmax"]) / 2.0 / width
    cy = (ann["ymin"] + ann["ymax"]) / 2.0 / height
    w = (ann["xmax"] - ann["xmin"]) / width
    h = (ann["ymax"] - ann["ymin"]) / height
    return f"{CLASS_INDEX[ann['class']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", type=Path, default=Path("model-training/datasets/neu-det-mirror"))
    parser.add_argument("--out", type=Path, default=Path("model-training/datasets/neu-det-yolo"))
    args = parser.parse_args()

    mirror: Path = args.mirror
    out: Path = args.out
    assert mirror.exists(), f"mirror dir missing: {mirror}"

    samples: list[dict] = []
    ann_folder_for = {"IMAGES": "ANNOTATIONS", "Validation_Images": "Validation_Annotations"}
    for folder in ("IMAGES", "Validation_Images"):
        img_dir = mirror / folder
        for img in sorted(img_dir.glob("*.jpg")):
            xml = (mirror / ann_folder_for[folder] / img.stem).with_suffix(".xml")
            if not xml.exists():
                raise FileNotFoundError(f"annotation missing for {img}")
            samples.append({"image": img, "xml": xml})

    if len(samples) != 1800:
        print(f"[warn] expected 1800 NEU-DET samples, found {len(samples)}", file=sys.stderr)

    parsed = []
    for s in samples:
        anns = parse_xml(s["xml"])
        dominant = Counter(a["class"] for a in anns).most_common(1)[0][0]
        parsed.append({**s, "anns": anns, "dominant": dominant})

    rng = random.Random(SEED)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for p in parsed:
        by_class[p["dominant"]].append(p)

    split_assign: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for cls, items in sorted(by_class.items()):
        rng.shuffle(items)
        n_train = int(round(len(items) * SPLIT_RATIOS["train"]))
        n_val = int(round(len(items) * SPLIT_RATIOS["val"]))
        split_assign["train"].extend(items[:n_train])
        split_assign["val"].extend(items[n_train : n_train + n_val])
        split_assign["test"].extend(items[n_train + n_val :])

    for split in split_assign:
        split_assign[split].sort(key=lambda p: p["image"].name)

    # leakage check: image-level disjointness
    seen: dict[str, str] = {}
    for split, items in split_assign.items():
        for p in items:
            key = p["image"].name
            if key in seen and seen[key] != split:
                raise RuntimeError(f"leak detected: {key} in both {seen[key]} and {split}")
            seen[key] = split

    if out.exists():
        shutil.rmtree(out)
    for split in split_assign:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)

    manifest_files = []
    for split, items in split_assign.items():
        for p in items:
            dst_img = out / split / "images" / p["image"].name
            shutil.copy2(p["image"], dst_img)
            label = "\n".join(to_yolo(a, 200, 200) for a in p["anns"])
            (out / split / "labels" / p["image"].stem).write_text(label + "\n", encoding="utf-8")
            manifest_files.append(
                {"split": split, "image": str(dst_img), "sha256": sha256_file(dst_img), "boxes": len(p["anns"])}
            )

    data_yaml = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    (out / "data.yaml").write_text(
        "\n".join(f"{k}: {v!r}" if k == "path" else f"{k}: {v}" for k, v in data_yaml.items()) + "\n",
        encoding="utf-8",
    )

    distribution = {}
    for split, items in split_assign.items():
        distribution[split] = dict(sorted(Counter(p["dominant"] for p in items).items()))

    provenance = {
        "dataset": "NEU-DET",
        "samples_total": len(parsed),
        "image_size": [200, 200],
        "classes": CLASSES,
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "split_counts": {k: len(v) for k, v in split_assign.items()},
        "class_distribution": distribution,
        "mirror_source": MIRROR_SOURCE,
        "original_source": ORIGINAL_SOURCE,
        "license_note": LICENSE_NOTE,
        "files": manifest_files,
        "data_yaml": str(out / "data.yaml"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"samples: {len(parsed)}")
    print(f"split counts: {provenance['split_counts']}")
    print("class distribution per split:")
    for split, dist in distribution.items():
        print(f"  {split}: {dist}")
    print("leakage check: PASS (image-level disjointness verified)")
    print(f"provenance written to {out / 'provenance.json'}")


if __name__ == "__main__":
    main()
