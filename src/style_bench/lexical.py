import string

from scipy import stats
from tqdm import tqdm
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .models import LexicalMetrics, WordLength, Richness, Legomena
from .config import LexicalConfig


class LexicalComputer:
    def __init__(self, config: LexicalConfig):
        # Download required NLTK data
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("averaged_perceptron_tagger_eng")
        nltk.download("averaged_perceptron_tagger")
        self.stop_words = set(stopwords.words("english"))
        self.config = config

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
            richness=Richness(ttr=[], mattr=[]),
            legomena=Legomena(hapax=[], dislegomena=[], trilegomina=[]),
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

            # Function words
            if self.config.function_words:
                function_word_frequency = self._calculate_function_word_frequency(
                    words_no_punct
                )
                lexical_metrics.function_word_frequency.append(function_word_frequency)

            # Word lengths
            if self.config.word_length:
                (
                    avg_word_length,
                    std_word_length,
                    skew_word_length,
                    kurtosis_word_length,
                ) = self._calculate_word_length(words_no_punct)
                lexical_metrics.word_length.avg.append(avg_word_length)
                lexical_metrics.word_length.std.append(std_word_length)
                lexical_metrics.word_length.skew.append(skew_word_length)
                lexical_metrics.word_length.kurtosis.append(kurtosis_word_length)

            # Richness
            if self.config.richness.mattr or self.config.richness.ttr:
                ttr, mattr = self._calculate_richness(words_no_punct)
                lexical_metrics.richness.ttr.append(ttr)
                lexical_metrics.richness.mattr.append(mattr)

            # Legomena
            if (
                self.config.legomena.hapax
                or self.config.legomena.dislegomena
                or self.config.legomena.trilegomina
            ):
                hapax, dis, tri = self._calculate_legomena(words_no_punct)
                lexical_metrics.legomena.hapax.append(hapax)
                lexical_metrics.legomena.dislegomena.append(dis)
                lexical_metrics.legomena.trilegomina.append(tri)

            # Sentiment
            if self.config.sentiment:
                sentiment = self._sentiment(text)
                lexical_metrics.sentiment.append(sentiment)

        return lexical_metrics

    # === Word Length ===
    def _calculate_word_length(self, words: list) -> tuple:
        """_summary_

        Args:
            words (list): _description_

        Returns:
            float: _description_
        """
        mean = np.mean([len(word) for word in words]) if words else 0
        std = np.std([len(word) for word in words]) if words else 0
        skew = stats.skew([len(word) for word in words]) if words else 0
        kurtosis = stats.kurtosis([len(word) for word in words]) if words else 0

        return float(mean), float(std), float(skew), float(kurtosis)

    # === Function Words ===
    def _calculate_function_word_frequency(self, words: list) -> float:
        """_summary_

        Args:
            words (list): _description_

        Returns:
            float: _description_
        """
        function_word_count = sum(1 for word in words if word in self.stop_words)
        return function_word_count / len(words) if words else 0

    # === Richness ===
    def _calculate_richness(self, words: list, window: int = 100) -> tuple:
        # TTR (Type-Token Ratio)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0

        # MATTR (Moving-Average Type-Token Ratio)
        # ensure window is not larger than the number of words
        if len(words) < window:
            mattr = ttr
        else:
            mattr = np.mean(
                [
                    len(set(words[i : i + window])) / window
                    for i in range(0, len(words), window)
                ]
            )

        return float(ttr), float(mattr)

    # === Legomena ===
    def _calculate_legomena(self, words: list) -> Legomena:
        """Calculate hapax, dislegomena, and trilegomina ratios"""
        word_counts = nltk.FreqDist(words)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        dislegomena = sum(1 for count in word_counts.values() if count == 2)
        trilegomina = sum(1 for count in word_counts.values() if count == 3)

        return (
            hapax / len(words) if words else 0,
            dislegomena / len(words) if words else 0,
            trilegomina / len(words) if words else 0,
        )

    def _sentiment(self, text: str) -> float:
        return None
