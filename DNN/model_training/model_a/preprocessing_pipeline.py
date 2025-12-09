# Model A: Data Preprocessing Pipeline
# Complete pipeline demonstrating all preprocessing steps
#
# This script shows how to use all Model A components together:
#   1. Clean text (remove URLs, HTML, normalize)
#   2. Extract features (emoji count, length, negations)
#   3. Tokenize (convert to word sequences)
#   4. Pad sequences (make all same length for batching)
#
# Run this file to see the complete pipeline in action:
#   python preprocessing_pipeline.py

from cleaning import TextCleaner, preprocess_for_sentiment
from tokenizer import SimpleTokenizer
from feature_engineering import FeatureExtractor
from embedding_prep import EmbeddingLoader
import pandas as pd
from typing import List, Dict


def run_preprocessing_example():
    """
    Example demonstrating the complete Model A preprocessing pipeline.
    Shows how all components work together.
    """
    
    # Sample texts for demonstration
    sample_texts = [
        "I LOVE this product!!! Best purchase ever 😊",
        "Not good at all. Very disappointed 😞",
        "This is okay, nothing special really...",
        "AMAZING quality! Highly recommend!!!",
        "Worst experience. Never buying again."
    ]
    
    print("=" * 60)
    print("MODEL A: DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Text Cleaning
    print("\n1. TEXT CLEANING")
    print("-" * 60)
    cleaner = TextCleaner(lowercase=True, remove_urls=True, remove_html=True)
    
    for i, text in enumerate(sample_texts, 1):
        cleaned = cleaner.clean(text)
        print(f"{i}. Original: {text}")
        print(f"   Cleaned:  {cleaned}\n")
    
    # Clean all texts
    cleaned_texts = cleaner.batch_clean(sample_texts)
    
    # Step 2: Feature Engineering
    print("\n2. FEATURE ENGINEERING")
    print("-" * 60)
    feature_extractor = FeatureExtractor()
    
    for i, text in enumerate(sample_texts, 1):
        features = feature_extractor.extract_all_features(text)
        print(f"{i}. Text: {text[:50]}...")
        print(f"   Features: {features}\n")
    
    # Step 3: Tokenization
    print("\n3. TOKENIZATION")
    print("-" * 60)
    tokenizer = SimpleTokenizer(max_vocab_size=1000, min_freq=1)
    
    # Build vocabulary from sample texts
    tokenizer.build_vocab(cleaned_texts)
    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
    
    # Encode texts to sequences
    for i, text in enumerate(cleaned_texts[:3], 1):
        tokens = tokenizer.tokenize(text)
        encoded = tokenizer.encode(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Tokens: {tokens}")
        print(f"   Encoded: {encoded}")
    
    # Step 4: Padding sequences to same length
    print("\n4. SEQUENCE PADDING")
    print("-" * 60)
    max_length = 20
    padded_sequences = tokenizer.encode_batch(cleaned_texts, max_length=max_length)
    
    for i, (text, seq) in enumerate(zip(cleaned_texts, padded_sequences), 1):
        print(f"{i}. {text[:40]:40} -> Length: {len(seq)}, Seq: {seq}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"- Processed {len(sample_texts)} texts")
    print(f"- Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"- Sequence length: {max_length}")
    print(f"- Ready for model training!")


if __name__ == "__main__":
    run_preprocessing_example()
