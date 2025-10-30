"""
example_pipeline.py

End-to-end example demonstrating the complete LSTM next-word prediction pipeline.

This script demonstrates:
1. Data preparation (cleaning and tokenization)
2. Sequence building (sliding windows)
3. Train/validation/test split
4. Model building (with optional GloVe embeddings)
5. Training with callbacks OR loading saved model to skip retraining
6. Performance evaluation
7. Text generation

Usage:
    python backend/example_pipeline.py

    # To retrain from scratch:
    python backend/example_pipeline.py --retrain

Note: This example uses a small synthetic dataset for quick demonstration.
For real applications, load your own text corpus using data_loader utilities.
"""
import numpy as np
import os
import sys

# Import our custom modules
from data_processing import texts_to_sequences
from sequence_builder import build_sliding_window_sequences, train_test_split
from lstm_model import build_lstm_model, load_glove_embeddings
from train_predict import train_and_save, save_training_artifacts, load_training_artifacts, generate_sentence
from performance_analysis import summarize_performance
from data_loader import load_sample_dataset, check_file_is_embeddings


def main():
    print("\n" + "="*80)
    print("LSTM Next-Word Prediction - Example Pipeline")
    print("="*80 + "\n")

    # Check if we should retrain or load saved model
    RETRAIN = "--retrain" in sys.argv
    SAVE_DIR = "trained_model"
    
    # Check if saved model exists
    model_exists = os.path.exists(SAVE_DIR) and os.path.exists(os.path.join(SAVE_DIR, "config.json"))
    
    if model_exists and not RETRAIN:
        print("🔄 Found saved model! Loading from disk to skip retraining...")
        print(f"   (Use --retrain flag to force retraining)\n")
        
        # Load saved artifacts
        model, tokenizer, config = load_training_artifacts(SAVE_DIR)
        WINDOW_SIZE = config["window_size"]
        vocab_size = config["vocab_size"]
        
        # Skip to step 7 (evaluation)
        skip_to_evaluation = True
    else:
        if RETRAIN:
            print("🔁 Retraining from scratch (--retrain flag detected)...\n")
        else:
            print("🆕 No saved model found. Training from scratch...\n")
        skip_to_evaluation = False

    if not skip_to_evaluation:
        # =====================================================================
        # 1. Prepare training data from Pride and Prejudice
        # =====================================================================
        print("Step 1: Loading training data from book...")
        
        # Load the actual book data
        from data_loader import load_text_from_file
        
        # Choose which book to train on
        # Option 1: Pride and Prejudice (1813) - Classic, formal English
        # Option 2: The Great Gatsby (1925) - Modern American English
        training_file = "training_data_gatsby.txt"  # Change to "training_data.txt" for Pride & Prejudice
        book_name = "The Great Gatsby"  # Change to "Pride and Prejudice" if using other file
        
        if not os.path.exists(training_file):
            print(f"❌ Error: Training file '{training_file}' not found!")
            print(f"   Please ensure {training_file} is in the backend directory.")
            print("\n📚 Available options:")
            print("   - training_data.txt (Pride and Prejudice - 1813)")
            print("   - training_data_gatsby.txt (The Great Gatsby - 1925)")
            sys.exit(1)
        
        # Load and split into sentences
        lines = load_text_from_file(training_file)
        full_text = '\n'.join(lines) if isinstance(lines, list) else lines
        
        # Split into sentences (simple split by period, question mark, exclamation)
        import re
        sentences = re.split(r'[.!?]+', full_text)
        # Clean up empty or very short sentences
        sample_texts = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        print(f"✓ Loaded {len(sample_texts)} sentences from {book_name}")
        print(f"  Training file: {training_file}")
        print(f"  Total characters: {len(full_text):,}")
        print(f"  Sample sentence: '{sample_texts[0][:80]}...'")
        
        # Optional: Check if you have the GloVe embedding file
        glove_path = "../wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
        use_glove = os.path.exists(glove_path)
        
        if use_glove:
            print(f"✓ Found GloVe embeddings at {glove_path}")
            # Verify it's actually embeddings (not training text)
            if check_file_is_embeddings(glove_path):
                print("  ✓ File confirmed as GloVe embedding format")
            else:
                print("  ⚠ Warning: File may not be in GloVe embedding format")
                use_glove = False
        else:
            print(f"ℹ GloVe embeddings not found at {glove_path}")
            print("  → Training embeddings from scratch")
            use_glove = False
    
        # =====================================================================
        # 2. Tokenize and create integer sequences
        # =====================================================================
        print("\nStep 2: Tokenizing and converting to integer sequences...")
        
        # Vocabulary size: balance between coverage and learnability
        # 2000-2500 = good balance (reduces <OOV> while maintaining accuracy)
        # 3000+ = maximum coverage but slower training
        NUM_WORDS = 2500  # Increased to reduce <OOV> tokens
        sequences, tokenizer = texts_to_sequences(
            texts=sample_texts,
            num_words=NUM_WORDS,
            oov_token="<OOV>",
            max_len=None,  # don't pad yet
        )
        
        vocab_size = min(len(tokenizer.word_index) + 1, NUM_WORDS)
        print(f"Vocabulary size: {vocab_size}")
        print(f"Total unique words found: {len(tokenizer.word_index)}")
        print(f"Sample sequence (first): {sequences[0][:10]}...")
    
        # =====================================================================
        # 3. Build sliding-window sequences
        # =====================================================================
        print("\nStep 3: Building sliding-window feature/label pairs...")
        
        WINDOW_SIZE = 7  # Increased to 7 for even more context
        X, y = build_sliding_window_sequences(
            sequences=sequences,
            window_size=WINDOW_SIZE,
            num_words=NUM_WORDS,
        )
        
        print(f"Generated {len(X)} training examples.")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
    
        # =====================================================================
        # 4. Split into train/validation/test
        # =====================================================================
        print("\nStep 4: Splitting data into train/validation/test sets...")
        
        # First split: 80% train+val, 20% test
        X_train_val, y_train_val, X_test, y_test = train_test_split(
            X, y, train_frac=0.8, shuffle=True, seed=42
        )
        
        # Second split: 80% of train_val -> train, 20% -> val
        X_train, y_train, X_val, y_val = train_test_split(
            X_train_val, y_train_val, train_frac=0.8, shuffle=True, seed=42
        )
        
        print(f"Train: {len(X_train)} examples")
        print(f"Validation: {len(X_val)} examples")
        print(f"Test: {len(X_test)} examples")
    
        # =====================================================================
        # 5. Build the LSTM model
        # =====================================================================
        print("\nStep 5: Building LSTM model...")
        
        # Optionally load GloVe embeddings if available
        embedding_matrix = None
        embedding_dim = 50  # default for training from scratch
        
        if use_glove:
            print("Loading GloVe embeddings (this may take a minute)...")
            try:
                embedding_dim = 50  # match the dimension in your GloVe file name
                embedding_matrix = load_glove_embeddings(
                    glove_path=glove_path,
                    word_index=tokenizer.word_index,
                    embedding_dim=embedding_dim,
                    vocab_size=vocab_size,
                )
                print(f"✓ Loaded GloVe embeddings: shape {embedding_matrix.shape}")
                trainable_embeddings = False  # freeze pretrained embeddings
            except Exception as e:
                print(f"⚠ Failed to load GloVe: {e}")
                print("  → Training embeddings from scratch instead")
                embedding_matrix = None
                trainable_embeddings = True
        else:
            trainable_embeddings = True
        
        model = build_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            input_length=WINDOW_SIZE,
            lstm_units=256,  # Increased to 256 for maximum capacity
            dense_units=128,  # Increased to 128 for better learning
            dropout_rate=0.2,  # Reduced dropout for more learning
            trainable_embeddings=trainable_embeddings,
        )
        
        print("\nModel architecture:")
        model.summary()
    
        # =====================================================================
        # 6. Train the model
        # =====================================================================
        print("\nStep 6: Training the model on Pride and Prejudice...")
        print("⏱️  This will take several minutes depending on your hardware...")
        
        history = train_and_save(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=100,  # Increased to 100 for more training
            batch_size=32,  # Smaller batch for better gradient estimates
            checkpoint_path="best_lstm_model.h5",
            final_model_path="final_lstm_model.h5",
            patience=10,  # Increased patience for convergence
        )
        
        print("\nTraining completed!")
        print(f"Final training accuracy: {history.history['sparse_categorical_accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
        
        # =====================================================================
        # 6.5. Save all training artifacts for future use (avoid retraining)
        # =====================================================================
        print("\nStep 6.5: Saving model + tokenizer + config for future use...")
        
        save_training_artifacts(
            model=model,
            tokenizer=tokenizer,
            window_size=WINDOW_SIZE,
            save_dir=SAVE_DIR,
            metadata={
                "embedding_dim": embedding_dim,
                "lstm_units": 256,  # Updated to match new architecture
                "used_glove": use_glove,
                "train_accuracy": float(history.history['sparse_categorical_accuracy'][-1]),
                "val_accuracy": float(history.history['val_sparse_categorical_accuracy'][-1]),
            }
        )
        
        # For evaluation, we'll need test data (if we just trained)
        # In load mode, we'll need to regenerate test data
        print("\n💾 Model saved! Next time you run this script, it will load instantly.")
    
    # =====================================================================
    # 7. Evaluate performance
    # =====================================================================
    print("\nStep 7: Evaluating model performance...")
    
    # If we loaded a saved model, we need to regenerate test data for evaluation
    if skip_to_evaluation:
        print("Regenerating test data from Pride and Prejudice...")
        from data_loader import load_text_from_file
        
        training_file = "training_data.txt"
        lines = load_text_from_file(training_file)
        full_text = '\n'.join(lines) if isinstance(lines, list) else lines
        import re
        sentences = re.split(r'[.!?]+', full_text)
        sample_texts = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        NUM_WORDS = config.get("num_words", vocab_size)
        sequences, _ = texts_to_sequences(
            texts=sample_texts,
            num_words=NUM_WORDS,
            oov_token="<OOV>",
            max_len=None,
        )
        X, y = build_sliding_window_sequences(
            sequences=sequences,
            window_size=WINDOW_SIZE,
            num_words=NUM_WORDS,
        )
        # Use a portion as test set
        _, _, X_test, y_test = train_test_split(X, y, train_frac=0.8, shuffle=True, seed=42)
        print(f"Test set: {len(X_test)} examples")
    
    # Build index_word mapping for predictions
    index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    
    summary = summarize_performance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        index_word=index_word,
        model_description="LSTM next-word prediction (toy example)",
    )
    
    # =========================================================================
    # 8. Generate some example text (Gatsby/Austen style based on training data)
    # =========================================================================
    print(f"\nStep 8: Generating text from seed phrases ({book_name} style)...")
    
    # Choose seed phrases based on the book
    if "gatsby" in training_file.lower():
        # The Great Gatsby style seed phrases
        seed_phrases = [
            "in my younger",
            "gatsby believed in",
            "the green light",
            "so we beat",
            "i was within",
        ]
    else:
        # Pride and Prejudice style seed phrases  
        seed_phrases = [
            "it is a",
            "mr darcy was",
            "elizabeth felt that",
            "the family were",
            "she could not",
        ]
    
    for seed in seed_phrases:
        generated = generate_sentence(
            model=model,
            seed_text=seed,
            tokenizer=tokenizer,
            window_size=WINDOW_SIZE,
            num_words=5,
            temperature=0.7,  # slightly more deterministic
        )
        print(f"Seed: \"{seed}\"")
        print(f"Generated: \"{generated}\"")
        print()
    
    print("="*80)
    print("Example pipeline completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
