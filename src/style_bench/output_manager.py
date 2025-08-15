# src/style_bench/output_manager.py


import yaml
import json
from datetime import datetime
from pathlib import Path
import pickle
from dataclasses import asdict


class OutputManager:
    def __call__(self, config, results, texts):
        """Main entry point to save results"""

        # Create output directory
        self.config = config
        self.results = results
        self.output_path = self._get_path()
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        self._save_metadata()

        # save texts
        self._save_texts(texts)

        # Save config
        self._save_config()

        # Save results
        self._save_results()

    def _get_path(self) -> str:
        path = Path(self.config.data.output_path)
        experiment_name = self.config.experiment_name.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = path / f"{experiment_name}_{timestamp}"
        return Path(output_path)

    def _save_metadata(self) -> None:
        metadata = {
            "experiment_name": self.config.experiment_name,
            "description": self.config.description,
            "timestamp": datetime.now().isoformat(),
        }
        metadata_path = self.output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _save_config(self) -> None:
        """Save the configuration used for the analysis"""
        config_path = self.output_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.dict(), f, default_flow_style=False)

    def _save_texts(self, texts: list[str]) -> None:
        """Save the original texts used for analysis"""
        texts_path = self.output_path / "texts.json"
        with open(texts_path, "w") as f:
            json.dump(texts, f, indent=4)

    def _save_results(self) -> None:
        # as a pickle
        results_path = self.output_path / "results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(self.results, f)

        # Save as JSON (convert dataclass to dict if necessary)
        results_json_path = self.output_path / "results.json"
        with open(results_json_path, "w") as f:
            try:
                # Try to convert dataclass to dict
                if hasattr(self.results, "__dataclass_fields__"):
                    json_data = asdict(self.results)
                else:
                    json_data = self.results
                json.dump(json_data, f, indent=4)
            except (TypeError, ValueError) as e:
                # If serialization fails, save a string representation
                json.dump(
                    {
                        "error": f"Could not serialize results: {str(e)}",
                        "results_str": str(self.results),
                    },
                    f,
                    indent=4,
                )
