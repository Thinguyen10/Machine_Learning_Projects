NLP React + FastAPI project

Overview

This repository is a port of a Streamlit sentiment-analysis demo into a React (Vite) frontend and a Python FastAPI backend. The backend wraps model training and prediction utilities that were originally implemented for the Streamlit app and uses TF-IDF preprocessing + either scikit-learn or Keras models. The frontend provides a simple UI for entering text, previewing preprocessing, running inference, and triggering training.

Repository layout

- backend/
  - main.py        # FastAPI application and endpoints
  - model.py       # model training/loading helpers and ModelWrapper
  - processing.py  # text preprocessing pipeline (clean, tokenize, TF-IDF)
  - training.py    # training wrappers
  - model.joblib   # (optional) saved sklearn model artifact
  - vect.joblib    # (optional) saved TF-IDF vectorizer
  - keras_model/   # (optional) Keras SavedModel directory

- frontend/
  - index.html
  - package.json
  - src/
    - App.jsx
    - main.jsx
    - components/   # React components mirroring Streamlit layout
    - services/api.js

Quick setup (macOS / zsh)

1) Python backend

- It's recommended to create a virtual environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

- Start the backend (development):

```bash
uvicorn backend.main:app --reload --port 8000
```

- Endpoints
  - GET /health — basic health check
  - GET /preview?csv_path=... — preview dataset rows
  - GET /transform?text=... — show preprocessing for a single text
  - GET /artifacts — list saved model/vectorizer artifacts
  - POST /train — JSON {"csv_path":"backend/data.csv","backend":"sklearn"}
  - POST /predict — JSON {"text":"Some text"}

Notes about training and dataset
- The training endpoint expects a CSV with at least columns `Body` and `Label` (same as original Streamlit dataset). Place the CSV at `backend/data.csv` or provide a full path in the `/train` request body.
- The original dataset was pulled from Kaggle. You can manually download it and place the CSV in `backend/` or I can add an upload endpoint so you can post the file from the frontend.
- Keras training requires TensorFlow installed in the Python environment (not included by default).

2) Frontend (Node + npm)

From the repo root:

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend to be available at http://localhost:8000 by default. The Vite dev server runs at http://localhost:3000.

Troubleshooting
- Module import issues: When running the backend with uvicorn use `uvicorn backend.main:app` (package import) — modules in `backend/` use package-qualified imports. If you see "No module named 'processing'" or similar, ensure you're running uvicorn from the repo root.
- Missing dependencies: If errors reference `nltk`, `tensorflow`, or `sklearn`, install them into the active venv. For quick local tests we added some lightweight fallbacks for NLTK tokenization in `backend/processing.py` but full behavior requires installing `nltk`.
- Training returns 400: usually because the CSV path isn't present or the CSV doesn't match expected columns. Check the path you pass to `/train` and verify the CSV format.

Next steps and improvements
- Add a file-upload endpoint to allow uploading the CSV from the React UI.
- Add progress notifications for long-running training (background task + websocket or polling).
- Improve frontend styling and make training UI match the Streamlit experience.

If you'd like, I can implement the upload endpoint and a small React file-input to upload a CSV and trigger training from the browser. Let me know which next step you prefer.
