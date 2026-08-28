#!/usr/bin/env python
"""Build a patient-disjoint subset of NIH ChestX-ray14 that fits in free storage.

    python scripts/build_xray_subset.py --source /content/nih_raw \
                                        --dest  /content/drive/MyDrive/CardioSense/data/xray/nih

Why this exists
---------------

ChestX-ray14 is ~45 GB across 112,120 images. A free Google Drive account holds
15 GB, and re-downloading 45 GB every Colab session is not a workflow. This script
selects the images the cardiomegaly pipeline would actually use and copies only
those, producing roughly 9 GB.

**This is a deviation from the full dataset and must be recorded as one.** It is
written as a script rather than notebook cells so that the exact cohort is
reproducible from a seed, and so the deviation can be described precisely in the
report rather than approximately.

What it selects
---------------

* **PA views only**, matching ``configs/xray_config.yaml``. Lateral films show a
  different projection of the heart and the model is not trained on them.
* **Every cardiomegaly-positive image** — the positive class is scarce (~2.5%
  prevalence) and discarding any of it would be indefensible.
* **Negatives subsampled by patient**, not by image, at a configurable ratio.

The by-patient rule matters more than it looks. A patient may contribute several
radiographs; sampling by image can place one of a patient's films in the subset
and another outside it, and — worse — the pipeline's own train/test split is
patient-level, so image-level sampling here would interact badly with it. Sampling
whole patients keeps the subset's patient structure intact.

What changes as a result
------------------------

The pipeline already subsamples negatives during training. Doing it here as well
means the **evaluation split is also drawn from the subset**, so reported metrics
describe this cohort rather than the full ChestX-ray14 test set. Prevalence in the
subset is far above the population rate by construction.

That is a real limitation, not a technicality, and the script writes a
``subset_manifest.json`` recording exactly what was selected so the claim can be
stated accurately.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

METADATA_CSV = "Data_Entry_2017_v2020.csv"
SPLIT_FILES = ("train_val_list.txt", "test_list.txt")


def build_subset(
    source: Path,
    dest: Path,
    negative_ratio: int = 8,
    view: str = "PA",
    target_label: str = "Cardiomegaly",
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Select and copy the subset.

    Args:
        source: Folder holding the full dataset — the metadata CSV, the split
            lists, and ``images/``.
        dest: Where the subset is written, in the same layout.
        negative_ratio: Negative patients per positive image. 8 matches the
            pipeline default.
        view: View position to keep.
        target_label: Finding to treat as positive.
        seed: Sampling seed, so the cohort is reproducible.
        dry_run: Report the selection without copying anything.

    Returns:
        A manifest describing the selection.

    Raises:
        FileNotFoundError: If the source layout is not as expected.
    """
    source, dest = Path(source), Path(dest)

    metadata_path = source / METADATA_CSV
    images_dir = source / "images"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{METADATA_CSV} not found in {source}. Extract the archives so that "
            "the CSV and images/ sit directly inside this folder."
        )
    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found in {source}.")

    frame = pd.read_csv(metadata_path)
    total_images = len(frame)

    frame = frame[frame["View Position"] == view].copy()
    after_view = len(frame)

    frame["target"] = frame["Finding Labels"].str.contains(target_label).astype(int)
    population_prevalence = float(frame["target"].mean())

    # Mirror `cardiosense.xray.data.build_target` exactly: keep every image
    # belonging to a patient who has ANY positive film, and sample negatives only
    # from patients with NO positive film.
    #
    # Matching the pipeline matters. If this script selected a different patient
    # set from the one the pipeline later chooses, training would look for images
    # that were never copied and die on FileNotFoundError.
    positive_patients = set(frame.loc[frame.target == 1, "Patient ID"])
    positive = frame[frame["Patient ID"].isin(positive_patients)]
    negative_only = frame[~frame["Patient ID"].isin(positive_patients)]

    negative_patients_all = negative_only["Patient ID"].unique()
    n_positive_images = int(frame.target.sum())
    target_negatives = negative_ratio * n_positive_images

    rng = np.random.default_rng(seed)
    images_per_patient = max(len(negative_only) / max(len(negative_patients_all), 1), 1.0)
    n_negative_patients = min(len(negative_patients_all),
                              int(np.ceil(target_negatives / images_per_patient)))
    keep = set(rng.choice(negative_patients_all, size=n_negative_patients,
                          replace=False)) if n_negative_patients else set()
    negative = negative_only[negative_only["Patient ID"].isin(keep)]

    subset = pd.concat([positive, negative]).sort_values("Image Index")
    overlap = set()   # disjoint by construction: negatives exclude positive patients

    manifest: dict[str, Any] = {
        "source": str(source),
        "destination": str(dest),
        "seed": seed,
        "view_filter": view,
        "target_label": target_label,
        "negative_ratio": negative_ratio,
        "images_in_full_dataset": total_images,
        "images_after_view_filter": after_view,
        "population_prevalence_pa_views": round(population_prevalence, 5),
        "n_positive": int(len(positive)),
        "n_negative": int(len(negative)),
        "n_subset": int(len(subset)),
        "n_negative_patients_sampled": int(n_negative_patients),
        "subset_prevalence": round(float(subset["target"].mean()), 5),
        "patients_on_both_sides": len(overlap),
        "estimated_size_gb": round(len(subset) * 0.000_08 * 1000 / 1000, 2),
        # Phase 2's prior correction needs the TRUE population prevalence, which
        # the filtered CSV no longer shows. Recorded here so it can be passed
        # explicitly to `python -m cardiosense.xray.calibrate --target-prevalence`.
        "population_prevalence_for_prior_correction": round(population_prevalence, 5),
        "training_command": ("python -m cardiosense.xray.train "
                             "--set dataset.negative_ratio=null"),
        "calibration_command": (f"python -m cardiosense.xray.calibrate "
                                f"--target-prevalence {population_prevalence:.5f}"),
        "caveat": (
            "Metrics computed on this subset describe this cohort, not the full "
            "ChestX-ray14 test set. Prevalence is inflated by construction "
            f"({subset['target'].mean():.3f} vs {population_prevalence:.3f} in PA "
            "views), so PR-AUC in particular is not comparable with published "
            "ChestX-ray14 numbers."
        ),
    }

    print(f"Full dataset      : {total_images:,} images")
    print(f"{view} views only      : {after_view:,}")
    print(f"Positive ({target_label}) : {len(positive):,} "
          f"({population_prevalence:.2%} of {view} views)")
    print(f"Negative sampled  : {len(negative):,} from {n_negative_patients:,} patients")
    print(f"Subset total      : {len(subset):,} images "
          f"({manifest['subset_prevalence']:.2%} positive)")
    if overlap:
        print(f"NOTE: {len(overlap)} patient(s) contribute both positive and "
              "negative films; this is expected and recorded in the manifest.")

    if dry_run:
        print("\n--dry-run: nothing copied.")
        return manifest

    (dest / "images").mkdir(parents=True, exist_ok=True)

    # The metadata CSV is FILTERED to the copied images, not copied wholesale.
    #
    # The pipeline reads this CSV and runs its own patient sampling over it. If it
    # listed all 112,120 images, the pipeline would select images that are not on
    # disk and training would fail. Filtering makes the CSV describe what is
    # actually present.
    #
    # The consequence: run training with `--set dataset.negative_ratio=null`,
    # because subsampling has already happened here and doing it twice shrinks the
    # cohort by another factor of eight.
    selected_names = set(subset["Image Index"])
    full_metadata = pd.read_csv(metadata_path)
    filtered = full_metadata[full_metadata["Image Index"].isin(selected_names)]
    filtered.to_csv(dest / METADATA_CSV, index=False)
    print(f"wrote filtered {METADATA_CSV} ({len(filtered):,} rows, "
          f"from {len(full_metadata):,})")

    # The official split lists are filtered too, for the same reason.
    for name in SPLIT_FILES:
        if not (source / name).exists():
            print(f"WARNING: {name} missing; the split will fall back to "
                  "grouped_random instead of the official patient-disjoint lists.")
            continue
        kept = [line.strip() for line in (source / name).read_text().splitlines()
                if line.strip() in selected_names]
        (dest / name).write_text("\n".join(kept) + "\n")
        print(f"wrote filtered {name} ({len(kept):,} entries)")

    print(f"\ncopying {len(subset):,} images...")
    missing = []
    for index, name in enumerate(subset["Image Index"], start=1):
        origin = images_dir / name
        if not origin.exists():
            missing.append(name)
            continue
        shutil.copy2(origin, dest / "images" / name)
        if index % 2000 == 0:
            print(f"  {index:,} / {len(subset):,}")

    if missing:
        print(f"\nWARNING: {len(missing)} selected images were not found in "
              f"{images_dir}. Not every archive was extracted.")
        manifest["missing_images"] = len(missing)

    manifest["images_copied"] = len(subset) - len(missing)

    with (dest / "subset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nSubset written to {dest}")
    print(f"Manifest         : {dest / 'subset_manifest.json'}")
    print("\n" + "=" * 70)
    print("TRAIN WITH IN-PIPELINE SUBSAMPLING DISABLED — it has already happened:")
    print("    python -m cardiosense.xray.train --set dataset.negative_ratio=null")
    print()
    print("PHASE 2 — pass the true population prevalence to the calibrator, because")
    print("the filtered CSV no longer shows it:")
    print(f"    python -m cardiosense.xray.calibrate "
          f"--target-prevalence {population_prevalence:.5f}")
    print("=" * 70)
    print("\nRecord this deviation in docs/datasets.md before reporting any metric "
          "from a model trained on it.")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a storage-constrained subset of NIH ChestX-ray14.")
    parser.add_argument("--source", required=True,
                        help="Folder containing the full dataset (CSV + images/).")
    parser.add_argument("--dest", required=True,
                        help="Where to write the subset.")
    parser.add_argument("--negative-ratio", type=int, default=8,
                        help="Negative patients per positive image (default 8).")
    parser.add_argument("--view", default="PA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the selection without copying.")
    args = parser.parse_args(argv)

    try:
        build_subset(Path(args.source), Path(args.dest),
                     negative_ratio=args.negative_ratio, view=args.view,
                     seed=args.seed, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
