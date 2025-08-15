from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class WordLength:
    avg: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    skew: List[float] = field(default_factory=list)
    kurtosis: List[float] = field(default_factory=list)


@dataclass
class Richness:
    ttr: List[float] = field(default_factory=list)
    mattr: List[float] = field(default_factory=list)


@dataclass
class Legomena:
    hapax: List[int] = field(default_factory=list)
    dislegomena: List[int] = field(default_factory=list)
    trilegomina: List[int] = field(default_factory=list)


@dataclass
class Sentiment:
    anger: List[float] = field(default_factory=list)
    disgust: List[float] = field(default_factory=list)
    fear: List[float] = field(default_factory=list)
    joy: List[float] = field(default_factory=list)
    neutral: List[float] = field(default_factory=list)
    sadness: List[float] = field(default_factory=list)
    surprise: List[float] = field(default_factory=list)


@dataclass
class LexicalMetrics:
    word_length: WordLength = field(default_factory=WordLength)
    function_word_frequency: List[float] = field(default_factory=list)
    richness: Richness = field(default_factory=Richness)
    legomena: Legomena = field(default_factory=Legomena)
    sentiment: Sentiment = field(default_factory=Sentiment)


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
