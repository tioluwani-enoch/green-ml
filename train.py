"""
Waste Sorting Model - Training Script
This script trains a CNN to classify waste into: compost, recycle, landfill
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 224  # MobileNetV2 expects 224x224 images
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Dataset path - you'll organize your images here
DATA_DIR = "data"  # Create this folder structure: data/train and data/val
CLASSES = ['compost', 'recycle', 'landfill']

# Model save path
MODEL_SAVE_PATH = "models/waste_sorting_model.keras"


def setup_data_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("data/train/compost", exist_ok=True)
    os.makedirs("data/train/recycle", exist_ok=True)
    os.makedirs("data/train/landfill", exist_ok=True)
    os.makedirs("data/val/compost", exist_ok=True)
    os.makedirs("data/val/recycle", exist_ok=True)
    os.makedirs("data/val/landfill", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("✓ Directory structure created!")
    print("\nNext step: Add your images to the data folders:")
    print("  - data/train/compost/  (your compost images)")
    print("  - data/train/recycle/  (your recycle images)")
    print("  - data/train/landfill/ (your landfill images)")
    print("  - data/val/compost/    (validation images)")
    print("  - data/val/recycle/    (validation images)")
    print("  - data/val/landfill/   (validation images)")


def create_data_generators():
    """Create image data generators with augmentation"""
    
    # Data augmentation for training (helps prevent overfitting)
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values to 0-1
        rotation_range=20,  # Randomly rotate images
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2,  # Randomly shift images vertically
        horizontal_flip=True,  # Randomly flip images
        zoom_range=0.2,  # Randomly zoom
        fill_mode='nearest'
    )
    
    # Only rescaling for validation (no augmentation)
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    
    return train_generator, val_generator


def create_model():
    """Create the CNN model using transfer learning with MobileNetV2"""
    
    # Load pre-trained MobileNetV2 (trained on ImageNet)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,  # Don't include the classification layer
        weights='imagenet'
    )
    
    # Freeze the base model (we'll just train the top layers)
    base_model.trainable = False
    
    # Build the complete model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),  # Prevent overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # 3 classes: compost, recycle, landfill
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history):
    """Visualize training progress"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("\n✓ Training history plot saved to models/training_history.png")


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("WASTE SORTING MODEL - TRAINING")
    print("=" * 60)
    
    # Setup directories
    setup_data_directories()
    
    # Check if data exists
    train_path = Path("data/train")
    if not any(train_path.iterdir()):
        print("\n⚠ Warning: No training data found!")
        print("Please add images to the data/train folders before training.")
        return
    
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    print(f"\n✓ Found {train_gen.samples} training images")
    print(f"✓ Found {val_gen.samples} validation images")
    print(f"✓ Classes: {train_gen.class_indices}")
    
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    
    # Create model
    model = create_model()
    model.summary()
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Plot results
    plot_training_history(history)
    
    # Final evaluation
    print("\nFinal Results:")
    print(f"  Best validation accuracy: {max(history.history['val_accuracy']):.2%}")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
    
    print("\n✓ Training complete! Next steps:")
    print("  1. Check models/training_history.png to see your training progress")
    print("  2. Run predict.py to test your model on new images")


if __name__ == "__main__":
    main()