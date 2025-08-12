import string
from collections import Counter

from dataclasses import dataclass
from typing import List, Dict

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


@dataclass
class TextMetrics:
    avg_word_length: float
    vocabulary_richness: float
    hapax_legomena_ratio: float
    avg_sentence_length: float
    pos_tag_distribution: Dict[str, float]
    sentence_length_variance: float
    paragraph_length_distribution: Dict[str, float]
    function_word_frequency: float
    flesch_kincaid_grade: float


@dataclass
class MetricsCollection:
    metric_names = [
        "avg_sentence_length",
        "avg_word_length",
        "vocabulary_richness",
        "hapax_legomena_ratio",
        "paragraph_length_distribution",
        "flesch_kincaid_grade",
        "function_word_frequency"
    ]

    avg_sentence_length: np.ndarray
    avg_word_length: np.ndarray
    vocabulary_richness: np.ndarray
    hapax_legomena_ratio: np.ndarray
    paragraph_length_distribution: np.ndarray
    flesch_kincaid_grade: np.ndarray
    function_word_frequency: np.ndarray


class StyleMetrics:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('averaged_perceptron_tagger')
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_text(self, text: str) -> dict:
        """
        Analyze text and return various stylometric features
        """
        # Basic text preparation
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_no_punct = [word for word in words if word not in string.punctuation]

        metrics = TextMetrics(
            avg_word_length=self._average_word_length(words_no_punct),
            vocabulary_richness=self._vocabulary_richness(words_no_punct),
            hapax_legomena_ratio=self._hapax_legomena_ratio(words_no_punct),
            avg_sentence_length=len(words) / len(sentences) if sentences else 0,
            pos_tag_distribution=self._pos_tag_distribution(words),
            sentence_length_variance=self._sentence_length_variance(sentences),
            paragraph_length_distribution=self._paragraph_length_distribution(text),
            function_word_frequency=self._function_word_frequency(words_no_punct),
            flesch_kincaid_grade=self._flesch_kincaid_grade(text)
        )
        
        return metrics
    
    def _average_word_length(self, words: list[str]) -> float:
        """Calculate average word length"""
        return np.mean([len(word) for word in words]) if words else 0
    
    def _vocabulary_richness(self, words: list[str]) -> float:
        """Calculate vocabulary richness (type-token ratio)"""
        return len(set(words)) / len(words) if words else 0
    
    def _hapax_legomena_ratio(self, words: list[str]) -> float:
        """Calculate ratio of words that appear only once"""
        word_counts = Counter(words)
        hapax = sum(1 for word, count in word_counts.items() if count == 1)
        return hapax / len(set(words)) if words else 0
    
    
    def _pos_tag_distribution(self, words: list[str]) -> dict[str, float]:
        """Calculate distribution of parts of speech"""
        pos_tags = nltk.pos_tag(words)
        tag_counts = Counter(tag for word, tag in pos_tags)
        total = sum(tag_counts.values())
        return {tag: count/total for tag, count in tag_counts.items()}
    
    def _sentence_length_variance(self, sentences: list[str]) -> float:
        """Calculate variance in sentence length"""
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        return np.var(lengths) if lengths else 0
    
    def _paragraph_length_distribution(self, text: str) -> dict[str, float]:
        """Analyze paragraph length distribution"""
        paragraphs = text.split('\n\n')
        lengths = [len(word_tokenize(para)) for para in paragraphs]
        return {
            'mean': np.mean(lengths) if lengths else 0,
            'variance': np.var(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'min': min(lengths) if lengths else 0
        }
    
    def _function_word_frequency(self, words: list[str]) -> float:
        """Calculate frequency of function words"""
        function_words = sum(1 for word in words if word in self.stop_words)
        return function_words / len(words) if words else 0
    
    def _flesch_kincaid_grade(self, text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0
            
        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        return 0.39 * (total_words/total_sentences) + 11.8 * (total_syllables/total_words) - 15.59
    
    def _count_syllables(self, word: str) -> int:
        """Rough syllable count for Flesch-Kincaid calculation. This is not good enough."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
            
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
            
        return count
    
    def _smooth_distribution(self, data: np.ndarray) -> np.ndarray:
        """Smooth the distribution using a Gaussian kernel"""
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 1000)

        return kde(x_range)
    
    def get_distributions(self, texts: list[str], smoothed=False) -> list[dict]:
        """Get stylometric metrics for a list of texts"""
        metrics_list = [self.analyze_text(text) for text in texts]

        style_metrics = MetricsCollection(
            avg_sentence_length=np.array([metric.avg_sentence_length for metric in metrics_list]),
            avg_word_length=np.array([metric.avg_word_length for metric in metrics_list]),
            vocabulary_richness=np.array([metric.vocabulary_richness for metric in metrics_list]),
            hapax_legomena_ratio=np.array([metric.hapax_legomena_ratio for metric in metrics_list]),
            paragraph_length_distribution=np.array([metric.paragraph_length_distribution['mean'] for metric in metrics_list]),
            flesch_kincaid_grade=np.array([metric.flesch_kincaid_grade for metric in metrics_list]),
            function_word_frequency=np.array([metric.function_word_frequency for metric in metrics_list])
        )

        return style_metrics
