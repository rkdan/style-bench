from dataclasses import dataclass
from typing import List, Dict

# This current configuration wont work because we have metrics and comparisons

@dataclass
class LexicalMetrics:
    avg_word_length: float
    function_word_frequency: float
    richness: float
    density: float
    levenshtein_distance: float

@dataclass
class SyntacticMetrics:
    pos_frequency: Dict[str, float]
    clauses: int
    dependency_distance: float

@dataclass
class LLMJudgeMetrics:
    straight_tell: str
    sample_comparison: str
    bert_score: float
    classifier_prediction: str