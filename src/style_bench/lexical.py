"""
Richness (MATTR, MTLD)
Word length
Function words
Density
Sentiment
"""

import string

from scipy import stats
from tqdm import tqdm
from .models import LexicalMetrics, WordLength

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class LexicalComputer:
    def __init__(self):
        # Download required NLTK data
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("averaged_perceptron_tagger_eng")
        nltk.download("averaged_perceptron_tagger")
        self.stop_words = set(stopwords.words("english"))

    def analyze_single(self, text: str) -> dict:
        """
        Analyze text and return various stylometric features
        """
        # Basic text preparation
        function_word_ratio = self._calculate_function_word_frequency(text)
        word_length = self._calculate_word_length(text)

        return (function_word_ratio, word_length)

    def analyze_corpus(self, texts: list[str], smoothed=False) -> LexicalMetrics:
        lexical_metrics = LexicalMetrics(
            word_length=WordLength(avg=[], std=[], skew=[], kurtosis=[]),
            function_word_frequency=[],
            richness=[],
            density=[],
            sentiment=[],
        )

        pbar = tqdm(
            texts,
            desc="🔍 Analyzing texts",
            unit="text",
            ncols=100,  # Width of progress bar
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour="green",
        )

        for text in pbar:
            words = word_tokenize(text.lower())
            words_no_punct = [word for word in words if word not in string.punctuation]

            function_word_frequency = self._calculate_function_word_frequency(
                words_no_punct
            )
            avg_word_length, std_word_length, skew_word_length, kurtosis_word_length = (
                self._calculate_word_length(words_no_punct)
            )
            richness = self._calculate_richness(words_no_punct)
            density = self._calculate_density(text)
            sentiment = self._sentiment(text)

            lexical_metrics.function_word_frequency.append(function_word_frequency)
            lexical_metrics.word_length.avg.append(avg_word_length)
            lexical_metrics.word_length.std.append(std_word_length)
            lexical_metrics.word_length.skew.append(skew_word_length)
            lexical_metrics.word_length.kurtosis.append(kurtosis_word_length)
            lexical_metrics.richness.append(richness)
            lexical_metrics.density.append(density)
            lexical_metrics.sentiment.append(sentiment)

        return lexical_metrics

    def _calculate_word_length(self, words: list) -> float:
        mean = np.mean([len(word) for word in words]) if words else 0
        std = np.std([len(word) for word in words]) if words else 0
        skew = stats.skew([len(word) for word in words]) if words else 0
        kurtosis = stats.kurtosis([len(word) for word in words]) if words else 0

        return mean, std, skew, kurtosis

    def _calculate_function_word_frequency(self, words: list) -> float:
        function_word_count = sum(1 for word in words if word in self.stop_words)
        return function_word_count / len(words) if words else 0

    def _calculate_richness(self, words: list) -> float:
        return None

    def _calculate_density(self, sentences: list) -> float:
        return None

    def _sentiment(self, text: str) -> float:
        return None
