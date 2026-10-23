"""
Severity threshold calibration.

The severity labels in this project are *derived* from annotated lesion coverage,
not independently annotated. The mild/moderate/severe cut points are therefore a
modelling choice that must be justified rather than asserted, otherwise the
severity head's reported accuracy is an artifact of wherever we happened to place
the thresholds.

This script reports, for a grid of candidate threshold pairs:
  - the resulting class balance per split
  - coverage agreement between train and valid (a threshold pair that splits one
    split one way and the other differently is a bad choice)

Run:
    python scripts/07_severity_calibration.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (PotatoLeafSegDataset,
                              coverage_to_grade)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("calib")

CANDIDATES = [
    (0.005, 0.03), (0.01, 0.05), (0.015, 0.06), (0.02, 0.08), (0.03, 0.10),
    (0.05, 0.15),
]


def coverages(dataset):
    return np.array([
        (dataset._rasterize(s, s["height"], s["width"]) > 0).mean()
        for s in dataset.samples
    ])


def main(args):
    covs = {}
    for split in args.splits:
        ds = PotatoLeafSegDataset(args.data_root, split=split)
        covs[split] = coverages(ds)
        logger.info("%s: %d images, coverage mean %.4f median %.4f",
                    split, len(covs[split]), covs[split].mean(),
                    np.median(covs[split]))

    logger.info("")
    logger.info("%-14s | %-28s | %s", "thresholds",
                "balance mild/mod/sev", " | ".join(f"{s:^22}" for s in args.splits))
    logger.info("-" * 100)

    rows = []
    for thresholds in CANDIDATES:
        line = f"{thresholds[0]:.3f}/{thresholds[1]:.3f}".ljust(14) + " | "
        cells = []
        stats = {}
        for split in args.splits:
            grades = np.array([coverage_to_grade(c, thresholds) for c in covs[split]])
            frac = [(grades == g).mean() for g in range(3)]
            stats[split] = frac
            cells.append("  ".join(f"{f * 100:5.1f}%" for f in frac))
        rows.append((thresholds, stats))

        # Split balance deviation from uniform 33/33/33, averaged across splits.
        imbalance = np.mean([
            np.abs(np.array(stats[s]) - 1 / 3).sum() for s in args.splits
        ])
        # Train/valid grade agreement, a direct proxy for threshold stability.
        grades_tr = np.array([coverage_to_grade(c, thresholds) for c in covs[args.splits[0]]])
        grades_va = np.array([coverage_to_grade(c, thresholds) for c in covs[args.splits[1]]])
        agree = np.mean([
            np.mean(coverage_to_grade(covs[args.splits[0]][i], thresholds) ==
                    coverage_to_grade(covs[args.splits[1]][i], thresholds))
            for i in range(min(len(grades_tr), len(grades_va)))
        ])

        line += "  ".join(c.ljust(22) for c in cells)
        line += f" | imb {imbalance:.3f}  agree {agree:.3f}"
        logger.info("%s", line)

    logger.info("")
    logger.info("Current setting in dataset.py is SEVERITY_THRESHOLDS = (0.01, 0.05).")
    logger.info("Pick the pair with the lowest imbalance that keeps agreement high.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",
                   default=str(PROJECT_ROOT / "potato-leaf-disease-segmentation-4g26g"))
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    a = p.parse_args()
    main(a)
