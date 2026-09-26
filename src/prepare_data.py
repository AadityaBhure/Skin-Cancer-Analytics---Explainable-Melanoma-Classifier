"""Prepare the guide's reproducible, balanced image-level HAM10000 split."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42
CLASSES = ["non_melanoma", "melanoma"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-cap", type=int, choices=[500, 1000], default=1000)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    raw = ROOT / "data/raw"
    metadata = raw / "HAM10000_metadata.csv"
    if not metadata.exists():
        metadata = raw / "HAM10000_metadata"
    if not metadata.is_file():
        raise SystemExit("Missing data/raw/HAM10000_metadata.csv. Place the HAM10000 metadata and JPEG folders in data/raw/.")
    frame = pd.read_csv(metadata)
    required = {"image_id", "lesion_id", "dx"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"Metadata must contain columns: {sorted(required)}")
    if frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise SystemExit("Metadata contains missing or duplicate image IDs.")
    known = {"mel", "akiec", "bcc", "bkl", "df", "nv", "vasc"}
    if not set(frame.dx).issubset(known):
        raise SystemExit("Metadata contains missing or unknown diagnosis labels.")
    images = {}
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            if path.stem in images:
                raise SystemExit(f"Duplicate image ID on disk: {path.stem}")
            images[path.stem] = path
    frame["source"] = frame.image_id.map(images)
    missing = int(frame.source.isna().sum())
    print(f"Metadata: {metadata.name}; missing images dropped: {missing}", flush=True)
    frame = frame.dropna(subset=["source"]).copy()
    readable = []
    for index, path in enumerate(frame.source, start=1):
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()
            readable.append(True)
        except (OSError, ValueError) as error:
            print(f"Skipping unreadable image {path.name}: {error}")
            readable.append(False)
        if index % 1000 == 0:
            print(f"Checked readability of {index}/{len(frame)} images.", flush=True)
    corrupt = len(readable) - sum(readable)
    frame = frame.loc[readable].copy()
    frame["class"] = np.where(frame.dx.eq("mel"), "melanoma", "non_melanoma")
    counts = frame["class"].value_counts()
    count = min(args.sample_cap, int(counts.get("melanoma", 0)))
    if count < 10 or counts.get("non_melanoma", 0) < count:
        raise SystemExit("Need at least 10 usable melanoma images and equally many non-melanoma images for stratified splitting.")
    balanced = pd.concat([frame[frame["class"].eq(name)].sample(n=count, random_state=SEED) for name in CLASSES])
    train, holdout = train_test_split(balanced, test_size=0.30, stratify=balanced["class"], random_state=SEED)
    val, test = train_test_split(holdout, test_size=0.50, stratify=holdout["class"], random_state=SEED)
    splits = {"train": train, "val": val, "test": test}
    for name, subset in splits.items():
        split_counts = subset["class"].value_counts()
        if len(split_counts) != 2 or split_counts.max() - split_counts.min() > 1:
            raise SystemExit(f"Invalid class balance for {name}.")
    processed = ROOT / "data/processed"
    # Check the exact resolved target before the only recursive deletion in this project.
    if processed.is_symlink() or processed.resolve() != ROOT / "data" / "processed":
        raise SystemExit("Refusing to clear a redirected processed directory.")
    if processed.exists():
        shutil.rmtree(processed)
    metrics = ROOT / "outputs/metrics"
    figures = ROOT / "outputs/figures"
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary, manifest = [], []
    for split, subset in splits.items():
        for name in CLASSES:
            destination = processed / split / name
            destination.mkdir(parents=True, exist_ok=True)
            selected = subset[subset["class"].eq(name)]
            summary.append({"split": split, "class": name, "count": len(selected)})
            print(f"{split}: {name} = {len(selected)}")
            for _, row in selected.iterrows():
                target = destination / f"{row.image_id}.jpg"
                shutil.copy2(row.source, target)
                manifest.append({"split": split, "image_id": row.image_id, "lesion_id": row.lesion_id,
                                 "class": name, "filename": target.relative_to(ROOT).as_posix()})
    summary = pd.DataFrame(summary)
    summary.to_csv(metrics / "dataset_summary.csv", index=False)
    pd.DataFrame(manifest).to_csv(metrics / "split_manifest.csv", index=False)
    overlap = {f"{a}_{b}": len(set(splits[a].lesion_id) & set(splits[b].lesion_id))
               for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    info = {"seed": SEED, "sample_cap": args.sample_cap, "images_per_class": count,
            "missing_images": missing, "unreadable_images": corrupt, "split_unit": "image",
            "shared_lesion_ids": overlap}
    (metrics / "dataset_preparation.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    ax = summary.pivot(index="split", columns="class", values="count").reindex(["train", "val", "test"]).plot.bar(rot=0, figsize=(8, 5))
    for container in ax.containers:
        ax.bar_label(container)
    ax.set(title="Balanced HAM10000 subset", xlabel="Split", ylabel="Image count")
    ax.legend(title="Class")
    ax.set_ylim(0, summary["count"].max() * 1.18)
    plt.tight_layout()
    plt.savefig(figures / "dataset_distribution.png", dpi=160)
    plt.close()
    print(f"Prepared {len(balanced)} images with seed {SEED}; sample cap {args.sample_cap} per class.")
    print(f"Shared lesion IDs across image-level splits: {overlap}. This is not lesion-independent evaluation.")


if __name__ == "__main__":
    main()
