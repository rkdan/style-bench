import pytest
import json
import tempfile
import os

from style_bench.utils import extract_texts


def test_extract_texts_valid_file():
    """Test basic functionality with valid JSON file"""
    data = [
        {"question": "What is AI?", "answer": "AI is artificial intelligence"},
        {"question": "How does ML work?", "answer": "ML uses algorithms"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        result = extract_texts(temp_path, "answer")
        assert result == ["AI is artificial intelligence", "ML uses algorithms"]
    finally:
        os.unlink(temp_path)


def test_extract_texts_missing_key():
    """Test that items missing the target key are skipped"""
    data = [
        {"question": "What is AI?", "answer": "AI is artificial intelligence"},
        {"question": "How does ML work?"},  # Missing answer key
        {"question": "What is data?", "answer": "Data is information"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        result = extract_texts(temp_path, "answer")
        assert result == ["AI is artificial intelligence", "Data is information"]
    finally:
        os.unlink(temp_path)


def test_extract_texts_file_not_found():
    """Test validation for non-existent file"""
    with pytest.raises(ValueError, match="File not found"):
        extract_texts("nonexistent.json", "answer")


def test_extract_texts_invalid_json():
    """Test validation for malformed JSON"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"invalid": json}')  # Invalid JSON
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid JSON file"):
            extract_texts(temp_path, "answer")
    finally:
        os.unlink(temp_path)


def test_extract_texts_not_list():
    """Test validation when JSON is not a list"""
    data = {"not": "a list"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="JSON file must contain a list"):
            extract_texts(temp_path, "answer")
    finally:
        os.unlink(temp_path)


def test_extract_texts_empty_list():
    """Test validation for empty list in JSON"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([], f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="JSON file contains an empty list"):
            extract_texts(temp_path, "answer")
    finally:
        os.unlink(temp_path)


def test_extract_texts_no_matching_keys():
    """Test when no items have the target key"""
    data = [{"question": "What is AI?"}, {"question": "How does ML work?"}]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        result = extract_texts(temp_path, "answer")
        assert result == []
    finally:
        os.unlink(temp_path)


def test_extract_texts_different_target_key():
    """Test with different target key"""
    data = [{"speech": "Hello world"}, {"speech": "Goodbye world"}]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        result = extract_texts(temp_path, "speech")
        assert result == ["Hello world", "Goodbye world"]
    finally:
        os.unlink(temp_path)
