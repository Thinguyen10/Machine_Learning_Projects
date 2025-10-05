# data_processing.py
# This module handles loading and preprocessing of the Skin Cancer dataset
# Now includes image augmentation to improve model performance

from pathlib import Path
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_skin_cancer_data(dataset_root, img_size=(64, 64), batch_size=32):
    """
    Load skin cancer images using the CSV labels created in Step 1.6.
    Returns: train_generator, val_generator
    """
    dataset_root = Path(dataset_root)

    # Locate CSVs
    train_csv = dataset_root / "train_labels.csv"
    test_csv = dataset_root / "test_labels.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("CSV labels not found. Ensure Step 1.6 ran successfully.")

    # Read CSVs
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Ensure 'filepath' column is absolute path for Keras
    df_train['filepath'] = df_train['filepath'].apply(lambda x: str(dataset_root.parent / x))
    df_test['filepath'] = df_test['filepath'].apply(lambda x: str(dataset_root.parent / x))

    # Keras ImageDataGenerator (rescale pixel values)
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow from dataframe
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen