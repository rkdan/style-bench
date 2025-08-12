import json
from typing import List


def extract_texts(file_path: str, target_key: str) -> List[str]:
    """Extract texts from JSON file

    Args:
        file_path (str): Path to JSON file containing list of dictionaries
        target_key (str): Key to extract from each dictionary

    Returns:
        List[str]: List of extracted text values
    """
    try:
        with open(file_path, "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")

    if not isinstance(json_data, list):
        raise ValueError(f"JSON file must contain a list, got {type(json_data)}")

    if not json_data:
        raise ValueError("JSON file contains an empty list")

    return [item[target_key] for item in json_data if target_key in item]
