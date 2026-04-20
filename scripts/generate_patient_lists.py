"""Generate shuffled patient ID lists for each pathology.

Usage:
    python scripts/generate_patient_lists.py BASE_MIMIC [--seed SEED] [--output_dir PATH]
"""

import argparse
import os
import pickle
import random

from dataset.utils import load_hadm_from_file

PATHOLOGIES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def main():
    parser = argparse.ArgumentParser(description="Generate shuffled patient ID lists")
    parser.add_argument(
        "base_mimic",
        help="Path to MIMIC dataset directory",
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (defaults to base_mimic)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.base_mimic

    for pathology in PATHOLOGIES:
        hadm_info = load_hadm_from_file(
            f"{pathology}_hadm_info_first_diag", base_mimic=args.base_mimic
        )
        ids = list(hadm_info.keys())

        random.seed(args.seed)
        random.shuffle(ids)

        output_path = os.path.join(output_dir, f"{pathology}_shuffled_ids.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(ids, f)

        print(f"{pathology}: {len(ids)} patients -> {output_path}")


if __name__ == "__main__":
    main()
