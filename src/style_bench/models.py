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
class Richness:
    ttr: np.ndarray
    mattr: np.ndarray


@dataclass
class Legomena:
    hapax: np.ndarray
    dislegomena: np.ndarray
    trilegomina: np.ndarray


@dataclass
class Sentiment:
    anger: np.ndarray
    disgust: np.ndarray
    fear: np.ndarray
    joy: np.ndarray
    neutral: np.ndarray
    sadness: np.ndarray
    surprise: np.ndarray


@dataclass
class LexicalMetrics:
    word_length: WordLength
    function_word_frequency: np.ndarray
    richness: Richness
    sentiment: np.ndarray
    legomena: Legomena


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
