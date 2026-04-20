import os
import pickle
from typing import Any, Dict, Tuple

from loguru import logger

from dataset.utils import load_hadm_from_file
from evaluators import load_evaluator
from utils.logging import read_from_pickle_file

PATHOLOGIES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")

AGENT_FIELDS = ("Diagnosis", "Gracious Diagnosis", "Physical Examination", "Late Physical Examination",
    "Action Parsing", "Treatment Parsing", "Diagnosis Parsing", "Rounds", "Invalid Tools")
AGENT_COUNT_FIELDS = ("Unnecessary Laboratory Tests", "Unnecessary Imaging")

FULL_INFO_FIELDS = ("Diagnosis", "Gracious Diagnosis")

REFERENCE_KEYS = (
    ("Discharge Diagnosis", ""),
    ("ICD Diagnosis", []),
    ("Procedures ICD9", []),
    ("Procedures ICD10", []),
    ("Procedures Discharge", ""),
)


def get_experiment_name(run_name: str) -> str:
    if "ZeroShot" in run_name:
        return "ZeroShot"
    if "FULL_INFO_PLI_N_VANILLA" in run_name:
        return "PLI_N_VANILLA"
    raise ValueError(f"Unsupported run name: {run_name}.")


def get_pathology_from_run_name(run_name: str) -> str:
    for pathology in PATHOLOGIES:
        if pathology in run_name:
            return pathology
    raise ValueError(f"Could not get pathology from run name: {run_name}.")


def load_results(results_path: str) -> Dict[int, Any]:
    merged: Dict[int, Any] = {}
    for record in read_from_pickle_file(results_path):
        merged.update(record)
    return merged

def reference_tuple(hadm: dict) -> Tuple[Any, ...]:
    return tuple(hadm.get(k, default) for k, default in REFERENCE_KEYS)


def score_agent(result: Any, patho: str, hadm: dict) -> dict:
    evaluator = load_evaluator(patho)
    return evaluator._evaluate_agent_trajectory(
        prediction=result["output"],
        input=result["input"],
        reference=reference_tuple(hadm),
        agent_trajectory=result["intermediate_steps"],
    )


def score_full_info(result: Any, patho: str, hadm: dict) -> dict:
    if isinstance(result, dict):
        prediction_text = result.get("Diagnosis", "")
    else:
        prediction_text = result
    evaluator = load_evaluator(patho)
    return evaluator._evaluate_agent_trajectory(
        prediction="Final Diagnosis: " + prediction_text,
        input="",
        reference=reference_tuple(hadm),
        agent_trajectory=[],
    )


def summarize_fields(evals: Dict[int, dict], experiment_type: str) -> Dict[str, float]:
    n = len(evals)
    if n == 0:
        return {"n": 0}

    summary: Dict[str, float] = {"n": n}
    fields = AGENT_FIELDS if experiment_type == "ZeroShot" else FULL_INFO_FIELDS
    for field in fields:
        summary[field] = sum(e["scores"][field] for e in evals.values()) / n

    if experiment_type == "ZeroShot":
        for field in AGENT_COUNT_FIELDS:
            summary[field] = sum(len(e["answers"][field]) for e in evals.values()) / n

    return summary


def score_run_dir(run_dir: str, base_mimic: str) -> Dict[str, Any]:
    """Score one run directory. Writes {run_dir}/{run_name}_evals.pkl and
    {run_dir}/{run_name}_scores.pkl.
    
    Returns a dict with keys: run_name, pathology, experiment, summary (aggregated scores),
    n_missing_refs.
    """
    run_dir = run_dir.rstrip(os.sep)
    run_name = os.path.basename(run_dir)
    experiment = get_experiment_name(run_name)
    patho = get_pathology_from_run_name(run_name)

    results_path = os.path.join(run_dir, f"{run_name}_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results file: {results_path}")

    results = load_results(results_path)
    hadm_info = load_hadm_from_file(f"{patho}_hadm_info_first_diag", base_mimic=base_mimic)

    evals: Dict[int, dict] = {}
    missing = 0
    for hadm_id, result in results.items():
        if hadm_id not in hadm_info:
            logger.warning(f"Skipping {hadm_id}: not in {patho}_hadm_info_first_diag")
            missing += 1
            continue
        hadm = hadm_info[hadm_id]
        if experiment == "ZeroShot":
            evals[hadm_id] = score_agent(result, patho, hadm)
        else:
            evals[hadm_id] = score_full_info(result, patho, hadm)

    summary = summarize_fields(evals, experiment)

    evals_path = os.path.join(run_dir, f"{run_name}_evals.pkl")
    scores_path = os.path.join(run_dir, f"{run_name}_scores.pkl")
    with open(evals_path, "wb") as f:
        pickle.dump(evals, f)
    with open(scores_path, "wb") as f:
        pickle.dump(summary, f)

    logger.info(
        f"Scored {run_name}: n={summary.get('n', 0)} "
        f"Diagnosis={summary.get('Diagnosis', 0):.1%} "
        f"Gracious={summary.get('Gracious Diagnosis', 0):.1%}"
    )

    return {
        "run_name": run_name,
        "pathology": patho,
        "experiment": experiment,
        "summary": summary,
        "n_missing_refs": missing,
    }
