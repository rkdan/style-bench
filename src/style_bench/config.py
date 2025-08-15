import yaml
from typing import Optional
from pydantic import BaseModel, field_validator
from pathlib import Path


class SyntacticConfig(BaseModel):
    pos_frequency: bool = True
    clauses: bool = True
    dependency_distance: bool = True


class LegomenaConfig(BaseModel):
    hapax: bool = True
    dislegomena: bool = True
    trilegomina: bool = True


class RichnessConfig(BaseModel):
    mattr: bool = True
    mtld: bool = True


class SentimentConfig(BaseModel):
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    classes: int = 7
    batch_size: int = 64


class LexicalConfig(BaseModel):
    richness: RichnessConfig = RichnessConfig()
    word_length: bool = True
    function_words: bool = True
    density: bool = True
    legomena: LegomenaConfig = LegomenaConfig()
    sentiment: SentimentConfig = SentimentConfig()


class DataConfig(BaseModel):
    data_path: str
    target_key: str = "answer"
    output_path: str = "output/"

    @field_validator("data_path")
    def validate_data_path(cls, v: str) -> str:
        """Ensure data_path is a valid directory."""
        path = Path(v)

        # Check if file exists
        if not path.exists():
            raise ValueError(f"Data path {v} does not exist")

        # Check if it's a file (not directory)
        if not path.is_file():
            raise ValueError(f"Data path {v} is not a file")

        # Check if it's a JSON file
        if path.suffix.lower() != ".json":
            raise ValueError(f"Data path {v} must be a JSON file (.json)")

        return str(path)

    @field_validator("output_path")
    def create_output_path(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class AnalysisConfig(BaseModel):
    lexical: LexicalConfig = LexicalConfig()
    syntactic: SyntacticConfig = SyntacticConfig()
    data: DataConfig

    # Can be defined by the user
    experiment_name: str
    description: Optional[str] = None


def load_config(config_path: str) -> AnalysisConfig:
    """Load and validate configuration from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return AnalysisConfig(**config_dict)
