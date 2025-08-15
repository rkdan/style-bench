"""all analysis happens in this script"""

# src/style_bench/scripts/analyze.py
import click
from loguru import logger
from style_bench.config import load_config
from style_bench.logging import setup_logging
from style_bench.utils import extract_texts
from style_bench.lexical import LexicalComputer
from style_bench.output_manager import OutputManager


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
def main(config_path, log_level, log_file):
    """Run stylometric analysis"""

    setup_logging(level=log_level, log_file=log_file)

    # == Load config ===
    try:
        logger.info("Loading config from {}", config_path)
        config = load_config(config_path)
        logger.success("Config loaded successfully!")
        logger.info("Experiment: {}", config.experiment_name)
        logger.info("Data path: {}", config.data.data_path)
        logger.info("Output path: {}", config.data.output_path)
        print("")

    except Exception as e:
        logger.error("Failed to load config: ", e)
        raise

    # == Data ingestion ===
    try:
        logger.info("Extracting texts from data file: {}", config.data.data_path)
        texts = extract_texts(config.data.data_path, config.data.target_key)
        logger.success("Extracted {} texts successfully!", len(texts))
    except Exception as e:
        logger.error("Failed to extract texts: ", e)
        raise

    # == Analysis tools loaded ===
    try:
        logger.info("Initializing lexical analysis tools")
        lexical_computer = LexicalComputer(config.lexical)
        logger.info("Lexical analysis tools initialized")
    except Exception as e:
        logger.error("Failed to initialize lexical analysis tools: ", e)
        raise

    # == Analysis execution ===
    try:
        logger.info("Starting lexical analysis on {} texts", len(texts))
        lexical_metrics = lexical_computer.analyze_corpus(
            texts,
        )
        logger.success("Lexical analysis completed")
    except Exception as e:
        logger.error("Failed to analyze texts: ", e)
        raise

    # == Output storage ===
    try:
        logger.info("Saving results to output path: {}", config.data.output_path)
        output_manager = OutputManager()
        output_manager(config, lexical_metrics, texts)
        logger.info("Results saved to {}", output_manager.output_path)
    except Exception as e:
        logger.error("Failed to save results: ", e)
        raise

    # == Summary ===
    logger.success("Analysis completed successfully!")


if __name__ == "__main__":
    main()
