# tests/test_config.py
import pytest
import tempfile
import json
import yaml
from pathlib import Path
from pydantic import ValidationError

from style_bench.config import (
    SyntacticConfig,
    RichnessConfig,
    LexicalConfig,
    DataConfig,
    AnalysisConfig,
    load_config,
)


class TestSyntacticConfig:
    """Test SyntacticConfig model"""

    def test_default_values(self):
        config = SyntacticConfig()
        assert config.pos_frequency is True
        assert config.clauses is True
        assert config.dependency_distance is True

    def test_custom_values(self):
        config = SyntacticConfig(
            pos_frequency=False, clauses=True, dependency_distance=False
        )
        assert config.pos_frequency is False
        assert config.clauses is True
        assert config.dependency_distance is False


class TestRichnessConfig:
    """Test RichnessConfig model"""

    def test_default_values(self):
        config = RichnessConfig()
        assert config.mattr is True
        assert config.mtld is True

    def test_custom_values(self):
        config = RichnessConfig(mattr=False, mtld=True)
        assert config.mattr is False
        assert config.mtld is True


class TestLexicalConfig:
    """Test LexicalConfig model"""

    def test_default_values(self):
        config = LexicalConfig()
        assert isinstance(config.richness, RichnessConfig)
        assert config.word_length is True
        assert config.function_words is True
        assert config.density is True
        assert config.sentiment is True

    def test_custom_richness(self):
        custom_richness = RichnessConfig(mattr=False, mtld=False)
        config = LexicalConfig(richness=custom_richness)
        assert config.richness.mattr is False
        assert config.richness.mtld is False

    def test_nested_richness_dict(self):
        # Test that we can pass richness as a dict
        config = LexicalConfig(richness={"mattr": False, "mtld": True})
        assert config.richness.mattr is False
        assert config.richness.mtld is True


class TestDataConfig:
    """Test DataConfig model including validators"""

    def test_missing_data_path(self):
        """Test that data_path is required"""
        with pytest.raises(ValidationError, match="Field required"):
            DataConfig()

    def test_valid_json_file(self):
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_json_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = DataConfig(data_path=temp_json_path, output_path=temp_dir)
                assert config.data_path == temp_json_path
                assert config.output_path == temp_dir
                # Check that output directory was created
                assert Path(temp_dir).exists()
        finally:
            Path(temp_json_path).unlink()

    def test_nonexistent_file(self):
        with pytest.raises(ValidationError, match="Data path .* does not exist"):
            DataConfig(data_path="nonexistent.json")

    def test_directory_instead_of_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError, match="Data path .* is not a file"):
                DataConfig(data_path=temp_dir)

    def test_non_json_file(self):
        # Create a temporary non-JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not json")
            temp_txt_path = f.name

        try:
            with pytest.raises(
                ValidationError, match="Data path .* must be a JSON file"
            ):
                DataConfig(data_path=temp_txt_path)
        finally:
            Path(temp_txt_path).unlink()

    def test_output_path_creation(self):
        # Create a temporary JSON file for data_path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_json_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_base:
                nested_output_path = Path(temp_base) / "nested" / "output" / "dir"

                config = DataConfig(  # noqa: F841
                    data_path=temp_json_path, output_path=str(nested_output_path)
                )

                # Check that nested directories were created
                assert nested_output_path.exists()
                assert nested_output_path.is_dir()
        finally:
            Path(temp_json_path).unlink()


class TestAnalysisConfig:
    """Test AnalysisConfig model"""

    @pytest.fixture
    def valid_json_file(self):
        """Create a temporary valid JSON file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            yield f.name
        Path(f.name).unlink()

    def test_minimal_config(self, valid_json_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AnalysisConfig(
                experiment_name="test_experiment",
                data={"data_path": valid_json_file, "output_path": temp_dir},
            )

            assert config.experiment_name == "test_experiment"
            assert config.description is None
            assert isinstance(config.lexical, LexicalConfig)
            assert isinstance(config.syntactic, SyntacticConfig)
            assert isinstance(config.data, DataConfig)

    def test_full_config(self, valid_json_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AnalysisConfig(
                experiment_name="test_experiment",
                description="Test description",
                lexical={
                    "richness": {"mattr": False, "mtld": True},
                    "word_length": False,
                    "sentiment": True,
                },
                syntactic={"pos_frequency": False, "clauses": True},
                data={"data_path": valid_json_file, "output_path": temp_dir},
            )

            assert config.experiment_name == "test_experiment"
            assert config.description == "Test description"
            assert config.lexical.richness.mattr is False
            assert config.lexical.word_length is False
            assert config.syntactic.pos_frequency is False

    def test_missing_experiment_name(self, valid_json_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError, match="experiment_name"):
                AnalysisConfig(
                    data={"data_path": valid_json_file, "output_path": temp_dir}
                )


class TestLoadConfig:
    """Test config loading from YAML files"""

    @pytest.fixture
    def valid_json_file(self):
        """Create a temporary valid JSON file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"text": "sample text", "speaker": "test"}], f)
            yield f.name
        Path(f.name).unlink()

    def test_load_valid_config(self, valid_json_file):
        # Create a valid YAML config
        config_data = {
            "experiment_name": "test_experiment",
            "description": "Test description",
            "lexical": {
                "richness": {"mattr": True, "mtld": False},
                "word_length": True,
                "sentiment": False,
            },
            "syntactic": {"pos_frequency": True, "clauses": False},
            "data": {"data_path": valid_json_file, "output_path": "test_output"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)

            assert config.experiment_name == "test_experiment"
            assert config.description == "Test description"
            assert config.lexical.richness.mattr is True
            assert config.lexical.richness.mtld is False
            assert config.lexical.word_length is True
            assert config.lexical.sentiment is False
            assert config.syntactic.pos_frequency is True
            assert config.syntactic.clauses is False
            assert config.data.data_path == valid_json_file

            # Check that output directory was created
            assert Path("test_output").exists()

        finally:
            Path(config_path).unlink()
            # Clean up created output directory
            if Path("test_output").exists():
                Path("test_output").rmdir()

    def test_load_nonexistent_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_invalid_yaml(self):
        # Create an invalid YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_load_invalid_config_data(self, valid_json_file):
        # Create YAML with invalid config (missing required fields)
        config_data = {
            "description": "Missing experiment_name",
            "data": {"data_path": valid_json_file, "output_path": "test_output"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ValidationError, match="experiment_name"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()


# Integration test
class TestConfigIntegration:
    """Test the whole config system together"""

    def test_realistic_config_workflow(self):
        # Create sample data file
        sample_data = [
            {"text": "Hello world", "speaker": "Obama"},
            {"text": "Good morning", "speaker": "Obama"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            data_file = f.name

        # Create realistic config
        config_data = {
            "experiment_name": "obama_lexical_analysis",
            "description": "Testing Obama's lexical features",
            "lexical": {
                "richness": {"mattr": True, "mtld": True},
                "word_length": True,
                "function_words": False,
                "density": True,
                "sentiment": True,
            },
            "syntactic": {
                "pos_frequency": True,
                "clauses": False,
                "dependency_distance": True,
            },
            "data": {"data_path": data_file, "output_path": "outputs/obama_test"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Load and validate config
            config = load_config(config_file)

            # Verify it loaded correctly
            assert config.experiment_name == "obama_lexical_analysis"
            assert config.lexical.richness.mattr is True
            assert config.lexical.function_words is False
            assert config.syntactic.clauses is False
            assert Path(config.data.output_path).exists()

        finally:
            # Cleanup
            Path(data_file).unlink()
            Path(config_file).unlink()
            if Path("outputs/obama_test").exists():
                Path("outputs/obama_test").rmdir()
                if Path("outputs").exists():
                    Path("outputs").rmdir()
