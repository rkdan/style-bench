import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from dataclasses import fields


def plot_distribution(
    ax, data, label, color, smoothed=True, bins=10, alpha_bar=0.3, alpha_fill=0.15
):
    """Generic function to plot a single distribution with optional KDE smoothing."""
    data = np.array(data)

    if smoothed:
        h, edges = np.histogram(data, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(
            centers,
            h,
            width=edges[1] - edges[0],
            align="center",
            alpha=alpha_bar,
            color=color,
        )

        # Add KDE curve
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        kde_values = kde(x_range)
        ax.plot(
            x_range,
            kde_values,
            color=color,
            linewidth=2,
            label=f"{label} (KDE)" if label else "KDE",
        )
    else:
        h, edges = np.histogram(data, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(
            centers,
            h,
            width=edges[1] - edges[0],
            align="center",
            alpha=0.7,
            color=color,
            label=label,
        )


def plot_multiple_distributions(
    data_dict,
    titles,
    xlabel,
    overall_title,
    figsize=None,
    smoothed=True,
    bins=10,
    colors=None,
):
    """
    Plot multiple distributions, optionally comparing different datasets.

    data_dict: {dataset_name: {metric_name: data_array}} or {metric_name: data_array} for single dataset
    titles: list of subplot titles
    xlabel: x-axis label
    overall_title: main title for the figure
    """
    if colors is None:
        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink"]

    # Handle single dataset case
    if not any(isinstance(v, dict) for v in data_dict.values()):
        data_dict = {"Dataset": data_dict}

    num_plots = len(titles)
    if figsize is None:
        figsize = (3 * num_plots, 3)

    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    metric_names = list(next(iter(data_dict.values())).keys())

    for i, (ax, title, metric) in enumerate(zip(axes, titles, metric_names)):
        ax.set_title(title)

        for j, (dataset_name, dataset_data) in enumerate(data_dict.items()):
            color = colors[j % len(colors)]
            label = dataset_name if len(data_dict) > 1 else None
            plot_distribution(ax, dataset_data[metric], label, color, smoothed, bins)

        ax.set_ylabel("Density" if smoothed else "Count")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        if len(data_dict) > 1:
            ax.legend()

    plt.suptitle(overall_title)
    plt.tight_layout()
    plt.show()


def plot_function_word_frequency(results_dict, smoothed=True, bins=10):
    """
    Plot function word frequency distributions.
    results_dict can be a single results object or {name: results_obj} for comparison
    """
    if not isinstance(results_dict, dict):
        results_dict = {"Dataset": results_dict}

    data_dict = {}
    for name, results in results_dict.items():
        data_dict[name] = {
            "function_word_frequency": np.array(results.function_word_frequency)
        }

    plot_multiple_distributions(
        data_dict,
        ["Function Word Frequency"],
        "Function Word Frequency",
        "Distribution of Function Word Frequencies",
        figsize=(4, 3),
        smoothed=smoothed,
        bins=bins,
    )


def plot_legomena(results_dict, smoothed=True, bins=10):
    """Plot legomena distributions with optional comparison."""
    if not isinstance(results_dict, dict):
        results_dict = {"Dataset": results_dict}

    data_dict = {}
    for name, results in results_dict.items():
        data_dict[name] = {
            "hapax": np.array(results.legomena.hapax),
            "dislegomena": np.array(results.legomena.dislegomena),
            "trilegomina": np.array(results.legomena.trilegomina),
        }

    plot_multiple_distributions(
        data_dict,
        ["Hapax Legomena", "Dislegomena", "Trilegomena"],
        "Legomena Ratio",
        "Distribution of Legomena",
        figsize=(9, 3),
        smoothed=smoothed,
        bins=bins,
    )


def plot_richness(results_dict, smoothed=True, bins=10):
    """Plot richness distributions with optional comparison."""
    if not isinstance(results_dict, dict):
        results_dict = {"Dataset": results_dict}

    data_dict = {}
    for name, results in results_dict.items():
        data_dict[name] = {
            "ttr": np.array(results.richness.ttr),
            "mattr": np.array(results.richness.mattr),
        }

    plot_multiple_distributions(
        data_dict,
        ["TTR", "MATTR"],
        "Richness Ratio",
        "Distribution of Richness Metrics",
        figsize=(6, 3),
        smoothed=smoothed,
        bins=bins,
    )


def plot_word_length(results_dict, smoothed=True, bins=10):
    """Plot word length distributions with optional comparison."""
    if not isinstance(results_dict, dict):
        results_dict = {"Dataset": results_dict}

    data_dict = {}
    for name, results in results_dict.items():
        data_dict[name] = {
            "avg": np.array(results.word_length.avg),
            "std": np.array(results.word_length.std),
            "skew": np.array(results.word_length.skew),
            "kurtosis": np.array(results.word_length.kurtosis),
        }

    plot_multiple_distributions(
        data_dict,
        ["Average Word Length", "Standard Deviation", "Skewness", "Kurtosis"],
        "Word Length Statistics",
        "Distribution of Word Length Metrics",
        figsize=(12, 3),
        smoothed=smoothed,
        bins=bins,
    )


def create_radar_chart_from_sentiment_multiple_log(
    sentiment_dict, title="Emotion Profile Comparison (Log Scale)", figsize=(6, 4)
):
    """
    Create radar chart directly from multiple Sentiment objects with log scale
    sentiment_dict: {'Dataset A': sentiment_obj1, 'Dataset B': sentiment_obj2}
    """
    # Get emotion names from the first sentiment object
    emotions = [
        field.name.capitalize() for field in fields(list(sentiment_dict.values())[0])
    ]

    # Number of emotions
    num_emotions = len(emotions)

    # Compute angles for each emotion (in radians)
    angles = np.linspace(0, 2 * np.pi, num_emotions, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink"]

    # Plot each dataset
    for i, (label, sentiment_obj) in enumerate(sentiment_dict.items()):
        # Calculate mean probability for each emotion by accessing attributes
        values = [
            np.mean(getattr(sentiment_obj, field.name))
            for field in fields(sentiment_obj)
        ]

        # Apply log transform
        epsilon = 1e-6  # Small value to avoid log(0)
        log_values = [np.log10(v + epsilon) for v in values]
        log_values_plot = log_values + log_values[:1]  # Close the plot

        color = colors[i % len(colors)]
        ax.plot(
            angles,
            log_values_plot,
            "o-",
            markersize=4,
            linewidth=2,
            markerfacecolor="white",
            color=color,
            label=label,
        )
        ax.fill(angles, log_values_plot, alpha=0.15, color=color)

    # Formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Log10(Mean Probability)", labelpad=20)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title(title, pad=20)

    plt.tight_layout()
    plt.show()


def plot_all_distributions(results_dict, smoothed=True, bins=10):
    """
    Plot all distribution types in sequence.
    results_dict can be a single results object or {name: results_obj} for comparison
    """
    print("Plotting Function Word Frequency...")
    plot_function_word_frequency(results_dict, smoothed=smoothed, bins=bins)

    print("Plotting Legomena...")
    plot_legomena(results_dict, smoothed=smoothed, bins=bins)

    print("Plotting Richness...")
    plot_richness(results_dict, smoothed=smoothed, bins=bins)

    print("Plotting Word Length...")
    plot_word_length(results_dict, smoothed=smoothed, bins=bins)

    print("Plotting Sentiment Radar Chart...")
    sentiment_data = {name: results.sentiment for name, results in results_dict.items()}
    create_radar_chart_from_sentiment_multiple_log(sentiment_data)
