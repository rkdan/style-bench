from dataclasses import dataclass
from typing import Dict

import numpy as np

# This current configuration wont work because we have metrics and comparisons


@dataclass
class WordLength:
    avg: np.ndarray
    std: np.ndarray
    skew: np.ndarray
    kurtosis: np.ndarray


@dataclass
class LexicalMetrics:
    word_length: WordLength
    function_word_frequency: np.ndarray
    richness: np.ndarray
    density: np.ndarray
    sentiment: np.ndarray


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
