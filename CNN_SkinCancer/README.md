# Skin Cancer Detection (CNN)

A small project that trains a Convolutional Neural Network (CNN) to classify dermoscopic skin images as benign or malignant. The repository provides a Streamlit user interface to download the dataset (via KaggleHub), load and preprocess images (with augmentation), build a CNN, and train/evaluate the model while visualizing training history.

Dataset

- Source: `jaiahuja/skin-cancer-detection` (Kaggle) — images derived from the ISIC archive.
- Approx. 2,357 images across multiple classes, including `benign keratosis`, `melanoma`, `nevus`, and other skin lesion types.


Project highlights

- Streamlit UI (`app.py`) for end-to-end interaction: download dataset, configure training hyperparameters, preview images, create model, and train.
- Data loading and augmentation using Keras `ImageDataGenerator` with a 20% validation split.
- Simple, transparent CNN architecture defined in `model.py` (3 convolutional blocks, a Dense layer, and softmax output).
- Training utilities and plotting helpers in `train.py` and `utils.py`.

Quick start

1. Create a Python environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. Run the Streamlit app (recommended):

```bash
streamlit run app.py
```

The Streamlit UI walks you through downloading the dataset automatically (via `kagglehub`), loading/preprocessing data, creating the CNN model and training it interactively.

If the KaggleHub download step fails, download the dataset manually from Kaggle, unzip it to a local folder, and use that folder path when calling `load_skin_cancer_data` from `data_processing.py` or by modifying the Streamlit logic to point to the local folder.

Files and responsibilities

- `app.py` — Main Streamlit application and UI controls (download dataset, set hyperparameters, run training).
- `streamlit_frontpage.py` — A simple informational front page shown at the top of the Streamlit app.
- `data_processing.py` — `load_skin_cancer_data(dataset_path, img_size=(64,64), batch_size=32)` uses `ImageDataGenerator` to provide `train` and `validation` generators. Includes augmentation (rotation, shift, shear, zoom, flips) and `validation_split=0.2`.
- `model.py` — `create_cnn_model(input_shape, num_classes, optimizer)` constructs and compiles the CNN:
  - Conv2D(32) + MaxPool
  - Conv2D(64) + MaxPool
  - Conv2D(128) + MaxPool
  - Flatten -> Dense(128, relu) -> Dense(num_classes, softmax)
- `train.py` — `train_and_evaluate(model, train_gen, val_gen, epochs=50)` compiles (categorical_crossentropy, Adam) and trains the model. Also evaluates on validation data.
- `utils.py` — `plot_training_history(history_dict)` plots training/validation loss and accuracy. The Streamlit app calls this to visualize results.
- `create_labels_csv.py` — Scans dataset folders (Train/Test) and 
  generates CSV label files (`train_labels.csv`, `test_labels.csv`) 
  listing image paths and corresponding class labels.
- `requirements.txt` — Python dependencies used by the project.

Configuration & hyperparameters (Streamlit sidebar)

- `Number of Epochs` (slider)
- `Batch Size` (16, 32, 64, 128)
- `Learning Rate` (0.1, 0.01, 0.001, 0.0001)
- `Optimizer` (Adam, SGD)
- `Image Size` (32, 64, 128) — a tuple `img_size = (size, size)` is used by the data loader

Outputs

- During training the Streamlit app shows live epoch metrics and a progress bar.
- Training history is persisted to `training_history.json` after a training session (if writable).
- The `utils.plot_training_history` function returns a Matplotlib `Figure` used for visualization.

Notes, tips and troubleshooting

- GPU/Memory: Training on large image sizes or big batches may exceed CPU/RAM/GPU limits. If you run out of memory, reduce `Image Size` and/or `Batch Size`.
- Validation split: The data loader uses `validation_split=0.2` in `ImageDataGenerator`. Ensure the dataset folder uses a structure suitable for `flow_from_directory` (class-subfolders like `dataset_path/benign/*` and `dataset_path/malignant/*`).
- Reproducibility: Results will vary with random augmentation and different optimizers/learning rates. For best results with limited data, try transfer learning from a pretrained backbone (not included in this repo).
- Kaggle credentials: `kagglehub` is used to download the dataset programmatically. If you prefer to use Kaggle's official CLI, follow Kaggle's docs to set up credentials and then place the dataset in the local `dataset_path`.

Suggested next steps / improvements

- Add a CLI entry point to run `train_and_evaluate` from the command line with explicit dataset path and hyperparameters.
- Add model checkpointing and early stopping to preserve the best model weights.
- Implement transfer learning with a pretrained model (MobileNetV2, EfficientNet, etc.) to improve performance.
- Add tests for data loader shape/labels and a small smoke test to ensure training loop runs on a tiny synthetic dataset.

Credits

- Dataset: Jai Ahuja — `jaiahuja/skin-cancer-detection` (Kaggle). Images originate from the ISIC archive.

License

- This repository is provided as-is for educational purposes. Adapt or relicense as you see fit.

Contact

- For questions or issues, open an issue in this repository.


