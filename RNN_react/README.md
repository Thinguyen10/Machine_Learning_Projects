# 🎬 LSTM Text Generation with React# 🎬 Multi-Scale Sentiment Analyzer - Class Activity



A full-stack web application for AI-powered text generation using LSTM neural networks, trained on F. Scott Fitzgerald's *The Great Gatsby*.## 📋 Table of Contents

- [Project Overview](#project-overview)

## 📋 Table of Contents- [What is React?](#what-is-react)

- [Project Overview](#project-overview)- [Learning Objectives](#learning-objectives)

- [Quick Start](#quick-start)- [Prerequisites](#prerequisites)

- [Project Structure](#project-structure)- [Part 1: Understanding the Project](#part-1-understanding-the-project)

- [Technology Stack](#technology-stack)- [Part 2: Setting Up Your Environment](#part-2-setting-up-your-environment)

- [Backend Architecture](#backend-architecture)- [Part 3: Running the Application Locally](#part-3-running-the-application-locally)

- [Model Configuration](#model-configuration)- [Part 4: Making Your Modifications](#part-4-making-your-modifications)

- [Troubleshooting OOV Tokens](#troubleshooting-oov-tokens)- [Part 5: Deploying to Vercel](#part-5-deploying-to-vercel)

- [Usage](#usage)- [Troubleshooting](#troubleshooting)

- [Additional Resources](#additional-resources)

---

---

## 🎯 Project Overview

## 🎯 Project Overview

This application uses a Many-to-One LSTM (Long Short-Term Memory) neural network to predict the next word in a sequence, enabling AI-powered text completion. The model learns the writing style of *The Great Gatsby* and generates text continuations from user-provided seed phrases.



### Key Features

- **LSTM Next-Word Prediction**: Trained on classic literature### What Does It Do?

- **Interactive Web Interface**: React-based UI with real-time generation

- **Adjustable Parameters**: Control creativity (temperature) and length

- **GloVe Embeddings**: 50-dimensional pretrained word vectors### Technology Stack

- **Model Persistence**: Automatic save/load to avoid retraining

- **Frontend:** React with Vite, Tailwind CSS

---- **Backend:** FastAPI (Python)

- **AI Model:** DistilBERT (Hugging Face Transformers)

## 🚀 Quick Start- **Deployment:** Vercel Cloud



### Prerequisites---

- Python 3.8+

- Node.js 16+## ⚛️ What is React?

- npm or yarn

### Understanding React

### 1. Train the LSTM Model

```bash**React** is a JavaScript library for building user interfaces, particularly for web applications. Think of it as a tool that helps you create interactive websites efficiently.

cd backend

pip install -r requirements.txt#### Key Concepts:

python pipeline.py --retrain

```1. **Components:** React apps are built from small, reusable pieces called components

*Training takes 15-25 minutes. Creates `trained_model/` directory with saved artifacts.*   - Like LEGO blocks - each component is a self-contained piece

   - Example: A button, a form, a navigation bar

### 2. Start the Backend Server

```bash2. **JSX:** A syntax that looks like HTML but works in JavaScript

cd backend   ```jsx

uvicorn main:app --reload --port 8000   const greeting = <h1>Hello, World!</h1>;

```   ```

*Backend runs at http://localhost:8000*

3. **State:** Data that can change in your app

### 3. Start the Frontend   - When state changes, React automatically updates what users see

```bash   - Example: The text in an input box, whether a button is clicked

cd frontend

npm install4. **Props:** How components pass data to each other

npm run dev   - Like function parameters

```   - Parent components send data to child components

*Frontend runs at http://localhost:5173*

#### Why React?

### 4. Use the Application

1. Open http://localhost:5173 in your browser- ✅ **Fast:** Only updates the parts of the page that change

2. Enter seed text (e.g., "in my younger")- ✅ **Modular:** Break complex UIs into simple, reusable components

3. Adjust parameters (10-70 words, temperature 0.1-2.0)# 🎬 Multi-Scale Sentiment Analyzer - Class Activity

4. Click "Generate Text" to see AI-powered completion

## 📋 Table of Contents

---- [Project Overview](#project-overview)

- [What is React?](#what-is-react)

## 📁 Project Structure- [Learning Objectives](#learning-objectives)

- [Prerequisites](#prerequisites)

```- [Part 1: Understanding the Project](#part-1-understanding-the-project)

RNN_react/- [Part 2: Setting Up Your Environment](#part-2-setting-up-your-environment)

├── backend/- [Part 3: Running the Application Locally](#part-3-running-the-application-locally)

│   ├── main.py                     # FastAPI server with 3 endpoints- [Part 4: Making Your Modifications](#part-4-making-your-modifications)

│   ├── pipeline.py                 # End-to-end training pipeline- [Part 5: Deploying to Vercel](#part-5-deploying-to-vercel)

│   ├── data_processing.py          # Text cleaning & tokenization- [Troubleshooting](#troubleshooting)

│   ├── sequence_builder.py         # Sliding window sequences- [Additional Resources](#additional-resources)

│   ├── lstm_model.py               # Model architecture & GloVe loading

│   ├── train_predict.py            # Training & text generation---

│   ├── performance_analysis.py     # Metrics (accuracy, perplexity)

│   ├── data_loader.py              # Utilities for loading training data## 🎯 Project Overview

│   ├── training_data_gatsby.txt    # The Great Gatsby training corpus

│   ├── requirements.txt            # Python dependenciesThis repository contains a React frontend and a FastAPI backend. The backend includes utilities and a tutorial-style pipeline for building a next-word prediction model (many-to-one RNN) using Keras, plus an existing DistilBERT-based sentiment analyzer.

│   └── trained_model/              # Saved model artifacts

│       ├── lstm_model.h5           # Keras model weights (5.7 MB)### Technology Stack

│       ├── tokenizer.pkl           # Tokenizer with vocabulary

│       └── config.json             # Model configuration- **Frontend:** React with Vite, Tailwind CSS

│- **Backend:** FastAPI (Python)

└── frontend/- **ML:** Transformers (Hugging Face) for sentiment; TensorFlow / Keras for LSTM next-word model

    ├── src/- **Deployment:** Vercel Cloud

    │   ├── App.jsx                 # Main app with intro & generation tabs

    │   ├── components/---

    │   │   ├── IntroPage.jsx       # Educational landing page

    │   │   ├── TextGenerationSection.jsx  # Text generation UI## ⚛️ What is React?

    │   │   ├── Header.jsx          # App header

    │   │   ├── InfoSection.jsx     # LSTM explanationReact is a JavaScript library for building user interfaces. It uses components, JSX, state and props to build interactive, reusable UI pieces.

    │   │   ├── InputSection.jsx    # Seed text input

    │   │   ├── ResultsSection.jsx  # Generated text display---

    │   │   └── ExamplesSection.jsx # Example seed phrases

    │   └── services/## 📚 Part 1: Understanding the Project

    │       └── api.js              # API client for backend

    ├── package.json### Project Structure (high-level)

    └── vite.config.js

``````

nlp-react/

---│

├── frontend/                    # React application

## 🛠 Technology Stack│   ├── src/

│   │   ├── components/          # Reusable UI components

### Frontend│   │   ├── services/            # API helpers

- **React 18** with Vite - Fast, modern UI framework│   │   └── App.jsx, main.jsx

- **Tailwind CSS** - Utility-first styling│   └── package.json

- **Lucide Icons** - Beautiful icon library│

├── backend/                     # FastAPI server and ML utilities

### Backend│   ├── main.py                  # API endpoints

- **FastAPI** - Modern Python web framework│   ├── model.py                 # DistilBERT sentiment analyzer

- **TensorFlow/Keras 2.12+** - Deep learning framework│   ├── data_processing.py       # Text cleaning + Keras Tokenizer wrapper

- **LSTM Architecture** - Recurrent neural network for sequence modeling│   ├── sequence_builder.py      # Sliding-window X/y builder and train/test split

- **GloVe Embeddings** - Pretrained word vectors (50-dim wiki_giga)│   ├── lstm_model.py            # LSTM model builder, GloVe loader, train helper

│   ├── train_predict.py         # Training pipeline and text generation

### Training Data│   ├── performance_analysis.py  # Metrics and performance evaluation

- **The Great Gatsby** by F. Scott Fitzgerald│   ├── example_pipeline.py      # End-to-end demo script

  - 299 KB, 3,478 sentences, 287,437 characters│   └── requirements.txt         # Python dependencies

  - 6,627 unique words│

  - Modern English (1925)├── vercel.json

└── README.md

---```



## 🧠 Backend Architecture### How the backend pipeline works (overview)



### 7 Core Modules1. Clean raw text and create integer sequences using `backend/data_processing.py`.

2. Convert integer sequences into sliding-window features and labels using `backend/sequence_builder.py`.

1. **data_processing.py** - Text preprocessing3. Build and train an LSTM model (optionally initialized with GloVe embeddings) using `backend/lstm_model.py`.

   - `remove_punctuation_and_split()` - Clean and tokenize text

   - `texts_to_sequences()` - Convert text to integer sequences using Keras Tokenizer## 🗂 Backend modules (data processing, sequencing, model, training, and analysis)

   - Handles OOV (Out-Of-Vocabulary) tokens with special `<OOV>` marker

This project includes five backend modules to prepare data, build, train, and analyze a next-word LSTM model:

2. **sequence_builder.py** - Training data preparation

   - `build_sliding_window_sequences()` - Create (X, y) pairs with sliding window- `backend/data_processing.py`

   - `train_test_split()` - Split data into train/validation/test sets- `backend/sequence_builder.py`

   - Example: Window size 7 → input=[w1,w2...w7], output=w8- `backend/lstm_model.py`

- `backend/train_predict.py`

3. **lstm_model.py** - Model architecture- `backend/performance_analysis.py`

   - `build_lstm_model()` - Construct LSTM with Embedding layer

   - `load_glove_embeddings()` - Initialize with pretrained vectorsBelow is a summary of what each file does, the key techniques and libraries used, and deeper notes about the model architecture.

   - `train_model()` - Train with ModelCheckpoint & EarlyStopping callbacks

   - Architecture: Embedding → Masking → LSTM(256) → Dense(128) → Softmax### `backend/data_processing.py`



4. **train_predict.py** - Training & inference- Purpose: Clean raw text and convert it into integer token sequences suitable for model training.

   - `train_and_save()` - Full training loop with model persistence- Main functions:

   - `load_trained_model()` - Load saved model from disk   - `remove_punctuation_and_split(texts: List[str]) -> List[List[str]]` — lowercases text, removes punctuation, and splits on whitespace to produce token lists.

   - `predict_next_word()` - Single-word prediction with top-k sampling   - `texts_to_sequences(texts: List[str], num_words: Optional[int], oov_token: Optional[str], max_len: Optional[int]) -> (sequences, tokenizer)` — uses Keras `Tokenizer` to map tokens to integers and optionally pads sequences.

   - `generate_sentence()` - Iterative text generation with temperature sampling- Libraries/techniques used:

   - TensorFlow / Keras `Tokenizer` and `pad_sequences` (from `tensorflow.keras.preprocessing`).

5. **performance_analysis.py** - Model evaluation   - Regular expressions for punctuation removal.

   - `compute_accuracy()` - Next-word prediction accuracy- Notes:

   - `compute_top_k_accuracy()` - Top-5 accuracy metric   - The Keras `Tokenizer` builds a `word_index` mapping when `fit_on_texts` is called. `texts_to_sequences` replaces tokens with their integer indices.

   - `compute_perplexity()` - Language model perplexity (lower = better)   - If `num_words` was passed to the tokenizer when it was created, you may want to filter or skip tokens with indices >= `num_words` during dataset construction.

   - `display_sample_predictions()` - Show model outputs with confidence scores

### `backend/sequence_builder.py`

6. **data_loader.py** - Data utilities

   - `load_text_from_file()` - Load training corpus from .txt files- Purpose: Turn integer token sequences into sliding-window feature/label pairs for next-word prediction and split into train/validation sets.

   - `load_sample_dataset()` - Built-in sample sentences for testing- Main functions:

   - `is_embeddings_file()` - Detect GloVe embedding files   - `build_sliding_window_sequences(sequences: List[List[int]], window_size: int, num_words: Optional[int]) -> (X, y)` — for each sequence produces examples like `words[i:i+n] -> words[i+n]`.

   - `train_test_split(X, y, train_frac=0.8, shuffle=True, seed=None)` — handy deterministic split with optional shuffling.

7. **pipeline.py** - End-to-end orchestration- Libraries/techniques used:

   - Coordinates all modules for complete training workflow   - NumPy for array manipulation and indexing.

   - Automatic model save/load with pickle- Notes:

   - `--retrain` flag to force retraining   - The sliding window greatly increases the number of training examples and usually improves RNN performance.

   - Generates sample text and performance reports   - When using a limited vocabulary (`num_words`), skip windows that reference out-of-range token ids to keep label indices valid.



### LSTM Model Architecture (In-Depth)### `backend/lstm_model.py`



**Many-to-One Sequence Model:**- Purpose: Build, initialize (with optional pretrained GloVe vectors), and train an LSTM-based next-word prediction model using Keras Sequential API.

```- Main functions:

Input: [word1, word2, word3, ..., word_n]  →  Output: word_{n+1}   - `build_lstm_model(...)` — constructs and compiles the model.

```   - `load_glove_embeddings(glove_path, word_index, embedding_dim=100, vocab_size=None)` — reads a GloVe file and creates an embedding matrix aligned with the tokenizer's `word_index`.

   - `cosine_similarity(vec_a, vec_b)` — utility to inspect embedding similarity.

**Layer Details:**   - `train_model(...)` — trains with `ModelCheckpoint` and `EarlyStopping` callbacks and returns training history.

- Libraries/techniques used:

1. **Embedding Layer** (vocab_size × 50)   - TensorFlow / Keras (Sequential API, layers, callbacks) and NumPy.

   - Maps word indices to 50-dimensional dense vectors   - GloVe pretrained embeddings (optional) — text file with word vectors (e.g. `glove.6B.100d.txt`).

   - Initialized with GloVe pretrained embeddings

   - `mask_zero=True` to ignore padding tokens### `backend/train_predict.py`

   - 125,000 parameters

- Purpose: Train the LSTM model with callbacks, save it, load trained models, and make next-word predictions or generate complete sentences.

2. **LSTM Layer** (256 units)- Main functions:

   - Captures long-range temporal dependencies   - `train_and_save(...)` — trains the model using `ModelCheckpoint` and `EarlyStopping` and saves the final trained model to disk.

   - `dropout=0.2`, `recurrent_dropout=0.2` for regularization   - `load_trained_model(model_path)` — loads a saved Keras model (.h5 or SavedModel format).

   - Returns single output vector (many-to-one)   - `predict_next_word(model, seed_sequence, index_word, top_k=1)` — given a sequence of word indices, returns the top-k most likely next words with probabilities.

   - 315,392 parameters   - `generate_sentence(model, seed_text, tokenizer, window_size, num_words=10, temperature=1.0)` — given a starting seed text, iteratively predicts and appends words to generate a sentence.

- Libraries/techniques used:

3. **Dense Layer** (128 units, ReLU)   - TensorFlow / Keras for model loading and prediction.

   - Additional representational capacity   - Temperature sampling for controllable text generation (lower temperature = more deterministic, higher = more random).

   - `Dropout(0.3)` to prevent overfitting- Notes:

   - 32,896 parameters   - `train_and_save` uses `ModelCheckpoint` to save the best model during training and `EarlyStopping` to halt training if validation loss stops improving.

   - `generate_sentence` uses the tokenizer to convert seed text to indices, then repeatedly predicts the next word and appends it to the context.

4. **Output Layer** (vocab_size, Softmax)

   - Probability distribution over vocabulary### `backend/performance_analysis.py`

   - Uses `sparse_categorical_crossentropy` loss

   - 322,500 parameters (for vocab=2500)- Purpose: Analyze and summarize the trained model's performance using metrics appropriate for next-word prediction.

- Main functions:

**Total Parameters:** ~795,788 (trainable: ~670,788)   - `compute_accuracy(model, X_test, y_test)` — overall next-word prediction accuracy (fraction of correct predictions).

   - `compute_top_k_accuracy(model, X_test, y_test, k=5)` — top-k accuracy (fraction where true word is in top k predictions).

---   - `compute_perplexity(model, X_test, y_test)` — perplexity metric (exp of average cross-entropy); lower is better.

   - `display_sample_predictions(...)` — shows sample inputs, true next word, and model predictions with confidence.

## ⚙️ Model Configuration   - `analyze_top_errors(...)` — identifies the most frequently mispredicted words.

   - `summarize_performance(...)` — generates a comprehensive report with all metrics, sample predictions, and a summary of the RNN's functioning and accuracy.

### Current Settings (Optimized for The Great Gatsby)- Libraries/techniques used:

   - NumPy for metric computation.

```json   - Perplexity as a language model evaluation metric (measures model uncertainty).

{- Notes:

  "window_size": 7,           // Input context length (7 words)   - For large vocabularies, traditional confusion matrices are impractical; instead we focus on accuracy, top-k accuracy, perplexity, and qualitative sample analysis.

  "vocab_size": 2500,         // Vocabulary size (top 2500 words)   - The `summarize_performance` function prints a detailed report and returns a dictionary with all metrics for programmatic use.

  "embedding_dim": 50,        // GloVe vector dimensions

  "lstm_units": 256,          // LSTM hidden units#### Model architecture (in-depth)

  "dense_units": 128,         // Dense layer size

  "dropout": 0.2,             // Dropout rate (LSTM)We implement a many-to-one LSTM network to predict the next word given a fixed-length input window of tokens.

  "dense_dropout": 0.3,       // Dropout rate (Dense)

  "epochs": 100,              // Max training epochs1) Embedding layer

  "batch_size": 32,           // Training batch size    - Purpose: Map discrete word indices into continuous vectors (dense representations) of size `embedding_dim` (100 in this project).

  "train_accuracy": "25-30%", // Expected accuracy    - Implementation: `Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=window_size, weights=[embedding_matrix], trainable=...)`

  "val_accuracy": "23-28%"    // Expected validation accuracy    - When `embedding_matrix` is supplied (from GloVe), the weights parameter initializes embeddings with pretrained vectors. If `trainable=False`, the embeddings remain fixed; otherwise they can be fine-tuned.

}    - Notes on indexing: Keras tokenizers commonly reserve index `0` for padding. Ensure row 0 of the embedding matrix is all zeros so padding maps to a zero vector.

```

2) Masking

### Training Stats    - Purpose: When sequences are padded to a common length, the model must ignore padded positions. The Masking mechanism tells downstream layers (like LSTM) which timesteps are padding.

- **Training Examples:** 29,484    - Implementation: either `mask_zero=True` in the `Embedding` layer or an explicit `Masking(mask_value=0.0)` layer. When `mask_zero=True`, the Embedding layer will automatically mask index 0.

- **Train/Val/Test Split:** 64% / 16% / 20%    - Note: If embeddings are trainable and you plan to update embeddings, masking still works, but be careful when using zero-vectors as meaningful tokens.

- **Training Time:** 15-25 minutes (CPU)

- **Model Size:** 5.7 MB (saved)3) LSTM layer

    - Purpose: LSTM (Long Short-Term Memory) is a recurrent architecture designed to capture long-range dependencies in sequences. It maintains a hidden state and an internal cell state, and uses gates (input, forget, output) to control information flow.

### Why These Settings?    - Key behaviors:

       - Learns temporal dependencies across the input window.

- **Vocab 2500**: Balances coverage (38% of unique words) vs. learnability       - With `return_sequences=False` (the default) the LSTM outputs a single vector summarizing the entire input sequence — appropriate for many-to-one tasks like next-word prediction.

  - Too small (1000) → 85% OOV tokens       - `dropout` controls input dropout, `recurrent_dropout` controls dropout on recurrent connections; these help regularize the model.

  - Too large (5000) → Poor accuracy, slow training

- **Window 7**: Enough context for coherent predictions4) Dense layer + Dropout

- **LSTM 256**: Sufficient capacity without overfitting    - Purpose: After the LSTM produces a single vector, a Dense (fully connected) layer with ReLU adds representational capacity before the final softmax. A Dropout layer reduces overfitting.

- **Dropout 0.2/0.3**: Regularization for single-book training

5) Output (softmax) layer

---    - Purpose: Produce a probability distribution across the vocabulary for the next-word prediction.

    - Implementation: `Dense(vocab_size, activation='softmax')`

## 🔧 Troubleshooting OOV Tokens    - Loss: `sparse_categorical_crossentropy` is used since labels are integer token ids (not one-hot vectors).



### Problem: Too Many `<OOV>` in Generated Text6) Optimization and callbacks

    - Optimizer: Adam (default hyperparameters) — adaptive optimizer that works well for RNNs.

If your output looks like this:    - Callbacks:

```       - `ModelCheckpoint` — saves the model weights with the best validation loss.

"gatsby <OOV> in <OOV> <OOV> and <OOV>"       - `EarlyStopping` — stops training when validation loss stops improving, optionally restoring best weights.

```

#### GloVe embeddings and their use

### Root Cause

**Vocabulary too small** - The model replaces unknown words with `<OOV>` during training, then learns to predict `<OOV>` as a common token.- GloVe (Global Vectors) provides pretrained static word vectors trained on large corpora. Using them can accelerate training and improve generalization when you have limited labeled data.

- Loading procedure:

### Solution Applied ✅   1. Read the GloVe file line-by-line: each line `word val1 val2 ... valN`.

 2. Map tokens from the tokenizer's `word_index` to vectors, inserting them into an `embedding_matrix` aligned with token indices.

**Increased NUM_WORDS from 1000 → 2500** in `backend/pipeline.py` (line 130) 3. Tokens not found remain zero vectors (so masked/padded as zeros); the Masking/`mask_zero` handling ensures the model ignores padded tokens.



### Vocabulary Trade-offs#### Practical notes



| Vocab Size | Coverage | OOV Rate | Accuracy | Training Time |- Reserve index 0 for padding. Keep row 0 in the embedding matrix zeroed.

|------------|----------|----------|----------|---------------|- If your tokenizer used `num_words` (a vocabulary cap), follow the same cap when building `vocab_size` and when filtering sliding windows.

| 1000 ❌    | 15%      | 85%      | ~27%     | Fast          |- Training time will depend on dataset size and hardware. Use a small subset to verify your pipeline before long runs.

| **2500** ✅ | 38%      | 62%      | ~25-30%  | Medium        |

| 3500       | 53%      | 47%      | ~23-28%  | Slow          |## ✅ Example high-level pipeline

| 5000       | 75%      | 25%      | ~20-25%  | Very Slow     |

1. Prepare raw text with `backend/data_processing.py` -> get integer sequences and `tokenizer`.

**2500 is the sweet spot** - Reduces OOV dramatically while maintaining accuracy.2. Build (X, y) with `backend/sequence_builder.py` using a chosen `window_size = n`.

3. Split into train/val/test sets using `train_test_split`.

### Alternative Solutions4. Optionally load GloVe via `lstm_model.load_glove_embeddings(...)` to create `embedding_matrix`.

5. Build the model with `lstm_model.build_lstm_model(vocab_size, embedding_dim=100, embedding_matrix=embedding_matrix, input_length=n, trainable_embeddings=False)`.

1. **Character-Level Model** - No OOV tokens ever (but slower, can create nonsense)6. Train with `train_predict.train_and_save(...)`, which uses `ModelCheckpoint` and `EarlyStopping` and saves the final model.

2. **Increase to 3500** - Better coverage (53%) but slower training7. Evaluate performance with `performance_analysis.summarize_performance(model, X_test, y_test, index_word)` to get accuracy, perplexity, and sample predictions.

3. **Filter OOV Post-Processing** - Remove `<OOV>` from output after generation8. Generate text with `train_predict.generate_sentence(model, seed_text, tokenizer, window_size, num_words=20)` to complete sentences.

4. **Larger Training Corpus** - Add more books by Fitzgerald for better coverage

## Dependencies

### To Retrain with New Vocabulary

```bashThe backend `requirements.txt` includes the Python packages required for these modules:

cd backend

python pipeline.py --retrain```text

```# in backend/requirements.txt

numpy>=1.24.0

**Expected Results:**tensorflow>=2.12.0

- Before: 80-90% of words are `<OOV>`# plus existing entries: fastapi, uvicorn, pydantic, torch, transformers, python-multipart

- After: 20-40% of words are `<OOV>````

- Accuracy: May drop ~2% but text quality improves significantly

If you're running on Apple Silicon you may prefer a Mac-optimized TensorFlow package (`tensorflow-macos`) or install the official TensorFlow build compatible with your machine.

---

---

## 📖 Usage

## 🚀 Running the Example Pipeline

### API Endpoints

To see a complete end-to-end demonstration of the LSTM next-word prediction pipeline, run:

**1. Check LSTM Status**

```bash```bash

GET http://localhost:8000/lstm/statuscd backend

```python example_pipeline.py

Response:```

```json

{This script demonstrates:

  "model_loaded": true,- Data preparation (cleaning and tokenization)

  "window_size": 7,- Sequence building with sliding windows

  "vocab_size": 2500,- Train/validation/test split

  "embedding_dim": 50- Model building and training with callbacks

}- Performance evaluation (accuracy, perplexity, sample predictions)

```- Text generation from seed phrases



**2. Generate Text**The example uses a small synthetic dataset for quick demonstration. For real applications, load your own text corpus (like the Pride and Prejudice text from Project Gutenberg) and adjust hyperparameters accordingly.

```bash

POST http://localhost:8000/lstm/generate---

Content-Type: application/json

## 🌐 Full-Stack LSTM Integration

{

  "seed_text": "in my younger",The project includes a complete web interface for the LSTM text generation model, integrated alongside the sentiment analyzer.

  "num_words": 30,

  "temperature": 0.7### Architecture Overview

}

```**Backend (FastAPI):**

Response:- `/lstm/status` - Get model status and configuration

```json- `/lstm/generate` - Generate text completion from seed text

{- `/lstm/examples` - Get example seed phrases

  "seed_text": "in my younger",

  "generated_text": "in my younger and more vulnerable years my father gave me some advice...",**Frontend (React):**

  "num_words_generated": 30,- Tab-based UI switching between Sentiment Analysis and Text Generation

  "temperature": 0.7- Real-time text generation with adjustable parameters

}- Model status monitoring

```- Example seed phrase library



**3. Get Example Seeds**### Running the Full Application

```bash

GET http://localhost:8000/lstm/examples1. **Train the LSTM Model (if not already trained):**

```   ```bash

   cd backend

### Web Interface   python example_pipeline.py

   ```

1. **Intro Page** - Explains LSTM architecture and how to use   This creates the `trained_model/` directory with:

2. **Text Generation Tab** - Interactive text generation with:   - `model.h5` - Trained LSTM weights

   - Seed text input (start with 2-3 words)   - `tokenizer.pkl` - Tokenizer for text preprocessing

   - Num words slider (10-70 words)   - `config.json` - Model configuration

   - Temperature slider (0.1-2.0):

     - **0.3-0.7**: Predictable, coherent (follows training closely)2. **Start the Backend Server:**

     - **0.8-1.2**: Balanced creativity   ```bash

     - **1.3-2.0**: Wild, experimental (more random)   cd backend

   uvicorn main:app --reload --port 8000

### Example Seeds from The Great Gatsby   ```

```   The server automatically loads the trained model on startup if available.

"in my younger"

"gatsby believed in"3. **Start the Frontend Development Server:**

"the green light"   ```bash

"so we beat on"   cd frontend

"old money and"   npm install  # First time only

"the city seen"   npm run dev

```   ```

   Access the application at `http://localhost:5173`

---

### Using the Text Generation Interface

## 🎓 Understanding the Model

1. Navigate to the "Text Generation" tab

### What is LSTM?2. Check the LSTM Status card - it should show "Loaded" if the model trained successfully

**Long Short-Term Memory** is a recurrent neural network architecture designed to remember patterns in sequential data. Unlike standard RNNs, LSTMs can capture long-range dependencies through:3. Enter seed text (e.g., "the quick brown") or click an example

- **Memory Cells** - Store information over time4. Adjust generation parameters:

- **Gates** - Control what to remember/forget:   - **Number of Words** (1-50): How many words to generate

  - Input Gate: What new information to store   - **Temperature** (0.1-2.0):

  - Forget Gate: What old information to discard     - Lower (0.3-0.7): More predictable, coherent text

  - Output Gate: What to output from memory     - Higher (1.0-2.0): More creative, random variations

5. Click "Generate Text" to see the completion

### How Next-Word Prediction Works

1. **Tokenization**: Convert text to integer sequences (word → ID)### API Examples

2. **Sliding Window**: Create training pairs ([w1,w2...w7] → w8)

3. **Embedding**: Map word IDs to dense vectors (using GloVe)**Check LSTM Status:**

4. **LSTM Processing**: Learn sequential patterns over 7-word context```bash

5. **Softmax**: Predict probability distribution over vocabularycurl http://localhost:8000/lstm/status

6. **Sampling**: Select next word (with temperature control)```

7. **Iteration**: Append predicted word, shift window, repeat

**Generate Text:**

### Why Accuracy is ~25-30%```bash

Next-word prediction is **fundamentally difficult** because:curl -X POST http://localhost:8000/lstm/generate \

- English has thousands of valid continuations for any phrase  -H "Content-Type: application/json" \

- Single books have limited vocabulary and patterns  -d '{

- Model must generalize from 29K examples to all possible 7-word contexts    "seed_text": "the quick brown",

    "num_words": 10,

**State-of-the-art models** (GPT-4, Claude) achieve 40-60% on this task with:    "temperature": 0.7

- 100B+ parameters (vs our 795K)  }'

- Trillions of training tokens (vs our 287K characters)```

- Multi-book training data

---

**Our 25-30% is excellent** for a single-book LSTM! 🎯

```

---

---

## 📚 Additional Notes


### Model Persistence
The trained model is automatically saved to `backend/trained_model/`:
- **lstm_model.h5** - Keras model weights
- **tokenizer.pkl** - Pickled tokenizer with word mappings
- **config.json** - Metadata (vocab size, window size, etc.)

Subsequent runs load the saved model instantly (no retraining needed).

### Force Retraining
```bash
python pipeline.py --retrain
```

### Dependencies
**Backend** (`backend/requirements.txt`):
```
fastapi>=0.100.0
uvicorn>=0.23.0
numpy>=1.24.0
tensorflow>=2.12.0
pydantic>=2.0.0
```

**Frontend** (`frontend/package.json`):
```
react@^18.2.0
vite@^4.4.0
tailwindcss@^3.3.0
lucide-react@^0.263.1
```

### Training on Different Books
1. Download text from [Project Gutenberg](https://www.gutenberg.org/)
2. Save as `backend/training_data_custom.txt`
3. Modify `pipeline.py` line 72-88 to load your file
4. Run `python pipeline.py --retrain`

---

## 🏆 Credits

- **Training Data**: *The Great Gatsby* by F. Scott Fitzgerald (1925) - [Project Gutenberg](https://www.gutenberg.org/ebooks/64317)
- **GloVe Embeddings**: 50-dimensional wiki_giga vectors
- **Frameworks**: TensorFlow/Keras, FastAPI, React
- **Architecture**: Many-to-One LSTM for next-word prediction

---

**Built as a class activity for CST-435 Deep Learning** 🎓
