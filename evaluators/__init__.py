from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator


def load_evaluator(pathology):
    if pathology == "appendicitis":
        return AppendicitisEvaluator()
    if pathology == "cholecystitis":
        return CholecystitisEvaluator()
    if pathology == "diverticulitis":
        return DiverticulitisEvaluator()
    if pathology == "pancreatitis":
        return PancreatitisEvaluator()
    raise NotImplementedError(f"No evaluator for pathology: {pathology}")
