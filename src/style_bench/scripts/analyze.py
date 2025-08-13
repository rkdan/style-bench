"""all analysis happens in this script"""

# src/style_bench/scripts/analyze.py
import click
from loguru import logger
from style_bench.config import load_config
from style_bench.logging import setup_logging


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

    except Exception as e:
        logger.error("Failed to load config: ", e)
        raise

    # == Data ingestion ===

    # == Analysis tools loaded ===

    # == Analysis execution ===

    # == Output storage ===

    logger.success("Analysis completed successfully!")


if __name__ == "__main__":
    main()
