# data_processing.py
# This module handles loading and preprocessing of the Skin Cancer dataset
# Now includes image augmentation to improve model performance

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_skin_cancer_data(dataset_path, img_size=(64,64), batch_size=32):
    """
    Load skin cancer dataset with training and validation generators.
    Apply image augmentation to training set to improve generalization.
    
    Arguments:
    - dataset_path: str, path to the root folder of dataset
    - img_size: tuple, resize images to this size
    - batch_size: int, number of images per batch
    
    Returns:
    - train_gen: generator for training images (with augmentation)
    - val_gen: generator for validation images (no augmentation)
    """

    # Training data generator with augmentation
    #image augmentation helps the CNN generalize better, since skin lesion datasets are usually small and imbalanced.
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # Normalize pixels to [0,1]
        rotation_range=40,           # Randomly rotate images up to 40 degrees to simulate different angles of skin lesions.
        width_shift_range=0.2,       # Random horizontal shift
        height_shift_range=0.2,      # Random vertical shift
        shear_range=0.2,             # SDistorts images to create variability.
        zoom_range=0.2,              # Random zoom
        horizontal_flip=True,        # Random horizontal flip
        vertical_flip=True,          # Random vertical flip
        fill_mode='nearest',         # Fill in missing pixels after transformations
        validation_split=0.2         # 20% data for validation
    )

    # Validation generator should NOT use augmentation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Training generator
    train_gen = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation generator
    val_gen = val_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen
