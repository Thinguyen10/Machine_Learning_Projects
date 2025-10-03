# train.py
# This module compiles, trains, and evaluates the CNN

def train_and_evaluate(model, train_gen, val_gen, epochs=50):
    """
    Compile and train the CNN model.
    
    Arguments:
    - model: Keras CNN model
    - train_gen: training data generator
    - val_gen: validation data generator
    - epochs: number of training iterations
    
    Returns:
    - history: training history object (loss & accuracy per epoch)
    """
    
    # Compile the model
    # Categorical crossentropy: suitable for multi-class classification
    # Adam optimizer: adaptive learning rate optimizer
    # Metrics: accuracy
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_gen,          # Training data
        epochs=epochs,      # Number of epochs
        validation_data=val_gen  # Validation data for evaluation
    )

    # Evaluate the model on validation set
    loss, accuracy = model.evaluate(val_gen)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return history
