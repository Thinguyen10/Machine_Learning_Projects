"""
data_loader.py

Utilities for loading training text data from various sources.

Functions:
- load_text_from_file: Load text from a single file.
- load_text_from_directory: Load all .txt files from a directory.
- load_sample_dataset: Load a built-in sample dataset for quick testing.

Notes:
- For LSTM next-word prediction, you need TEXT DATA (sentences/documents), not embedding vectors.
- GloVe files (like wiki_giga_2024_50_MFT20_vectors...) are EMBEDDINGS, not training data.
- Use these utilities to load your training corpus (books, articles, Wikipedia, etc.).
"""
from typing import List
import os
import glob


def load_text_from_file(filepath: str, encoding: str = "utf-8") -> List[str]:
    """
    Load text from a single file and split into sentences or lines.

    Args:
        filepath: Path to the text file.
        encoding: File encoding (default: utf-8).

    Returns:
        List of strings (sentences or lines).

    Notes:
        - This function splits on newlines. For sentence-level splitting, you may want to use
          nltk.sent_tokenize or similar.
    """
    with open(filepath, "r", encoding=encoding, errors="ignore") as f:
        lines = f.readlines()

    # Filter out empty lines
    texts = [line.strip() for line in lines if line.strip()]

    print(f"Loaded {len(texts)} lines from {filepath}")
    return texts


def load_text_from_directory(dirpath: str, pattern: str = "*.txt", encoding: str = "utf-8") -> List[str]:
    """
    Load all text files from a directory matching a pattern.

    Args:
        dirpath: Path to the directory.
        pattern: Glob pattern to match files (default: *.txt).
        encoding: File encoding (default: utf-8).

    Returns:
        List of strings (all lines from all files).
    """
    filepaths = glob.glob(os.path.join(dirpath, pattern))

    if not filepaths:
        print(f"Warning: No files matching '{pattern}' found in {dirpath}")
        return []

    all_texts = []
    for filepath in filepaths:
        texts = load_text_from_file(filepath, encoding=encoding)
        all_texts.extend(texts)

    print(f"Loaded {len(all_texts)} total lines from {len(filepaths)} files")
    return all_texts


def load_sample_dataset() -> List[str]:
    """
    Load a built-in sample dataset for quick testing.

    Returns:
        List of sample sentences.

    Notes:
        - This is a tiny corpus for demonstration only.
        - For real training, load a larger dataset using load_text_from_file or load_text_from_directory.
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Where there is a will, there is a way.",
        "Actions speak louder than words.",
        "The early bird catches the worm.",
        "Practice makes perfect.",
        "Time flies like an arrow.",
        "Knowledge is power.",
        "The pen is mightier than the sword.",
        "Beauty is in the eye of the beholder.",
        "When in Rome, do as the Romans do.",
        "A picture is worth a thousand words.",
        "Two heads are better than one.",
        "The grass is always greener on the other side.",
        "You can't judge a book by its cover.",
        "Every cloud has a silver lining.",
        "Better late than never.",
        "A stitch in time saves nine.",
    ]

    print(f"Loaded {len(sample_texts)} sample sentences")
    return sample_texts


def check_file_is_embeddings(filepath: str, max_lines: int = 5) -> bool:
    """
    Check if a file is a GloVe-format embedding file (not training text).

    Args:
        filepath: Path to the file to check.
        max_lines: Number of lines to check.

    Returns:
        True if file appears to be embeddings (word followed by many floats), False otherwise.

    Notes:
        - GloVe files have format: word float1 float2 ... floatN
        - Training text files have natural language sentences.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                parts = line.strip().split()
                if len(parts) < 10:
                    return False  # embeddings typically have 50-300 dimensions
                # Check if parts[1:] are mostly floats
                try:
                    float_count = sum(1 for p in parts[1:6] if _is_float(p))
                    if float_count < 4:
                        return False
                except:
                    return False
        return True
    except:
        return False


def _is_float(s: str) -> bool:
    """Helper to check if a string is a float."""
    try:
        float(s)
        return True
    except:
        return False


__all__ = [
    "load_text_from_file",
    "load_text_from_directory",
    "load_sample_dataset",
    "check_file_is_embeddings",
]
