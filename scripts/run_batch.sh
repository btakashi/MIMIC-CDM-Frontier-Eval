#!/usr/bin/env bash
# Run run.py or run_full_info.py for all four pathologies with a given model.
# Rerun with a larger MAX_PATIENTS to continue where you left off.
#
# Usage:
#   scripts/run_batch.sh SCRIPT MODEL RUN_ID MAX_PATIENTS
#
# Examples:
#   scripts/run_batch.sh run_full_info.py Gemini25FlashLite batch1 100
#   scripts/run_batch.sh run.py Gemini25FlashLite batch1 100

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 SCRIPT MODEL RUN_ID MAX_PATIENTS"
    echo "Examples:"
    echo "  $0 run_full_info.py Gemini25FlashLite batch1 100"
    echo "  $0 run.py Gemini25FlashLite batch1 100"
    exit 1
fi

SCRIPT="$1"
MODEL="$2"
RUN_ID="$3"
MAX_PATIENTS="$4"
DATA_DIR="../mimic-iv-cdm-dataset"

for pathology in appendicitis cholecystitis diverticulitis pancreatitis; do
    echo "Running $pathology with $MODEL via $SCRIPT (max_patients=$MAX_PATIENTS)"
    python "$SCRIPT" \
        model="$MODEL" \
        pathology="$pathology" \
        run_id="$RUN_ID" \
        patient_list_path="$DATA_DIR/${pathology}_shuffled_ids.pkl" \
        max_patients="$MAX_PATIENTS"
done
