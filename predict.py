"""
Waste Sorting Model - Prediction Script
Use this to classify new waste images after training
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Configuration
MODEL_PATH = "models/waste_sorting_model.keras"
IMG_SIZE = 224
CLASSES = ['compost', 'recycle', 'landfill']


def load_and_preprocess_image(image_path):
    """Load and preprocess an image for prediction"""
    
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img, img_array


def predict_image(model, image_path):
    """Predict the class of a single image"""
    
    # Load and preprocess
    original_img, processed_img = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASSES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    return original_img, predicted_class, confidence, predictions[0]


def visualize_prediction(image, predicted_class, confidence, all_predictions):
    """Display the image with prediction results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {predicted_class.upper()}\nConfidence: {confidence:.1%}', 
                  fontsize=14, fontweight='bold')
    
    # Display confidence bars
    colors = ['#4CAF50', '#2196F3', '#FF9800']  # Green, Blue, Orange
    bars = ax2.barh(CLASSES, all_predictions, color=colors)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_title('Classification Scores', fontsize=14)
    ax2.set_xlim([0, 1])
    
    # Add percentage labels
    for i, (bar, pred) in enumerate(zip(bars, all_predictions)):
        ax2.text(pred + 0.02, i, f'{pred:.1%}', va='center')
    
    plt.tight_layout()
    return fig


def predict_single_image(image_path):
    """Predict a single image and display results"""
    
    print(f"\nLoading model from {MODEL_PATH}...")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first by running: python train.py")
        return
    
    # Load model
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    print(f"\nPredicting class for: {image_path}")
    
    # Make prediction
    img, pred_class, confidence, all_preds = predict_image(model, image_path)
    
    # Display results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Category: {pred_class.upper()}")
    print(f"Confidence: {confidence:.1%}")
    print("\nAll Scores:")
    for class_name, score in zip(CLASSES, all_preds):
        print(f"  {class_name:10s}: {score:.1%}")
    print("=" * 50)
    
    # Visualize
    fig = visualize_prediction(img, pred_class, confidence, all_preds)
    
    # Save visualization
    output_path = f"predictions_{Path(image_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()


def predict_batch(folder_path):
    """Predict all images in a folder"""
    
    print(f"\nLoading model from {MODEL_PATH}...")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first by running: python train.py")
        return
    
    # Load model
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
    
    # Get all image files
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"\nFound {len(image_files)} images. Processing...\n")
    
    # Process each image
    results = []
    for img_path in image_files:
        try:
            _, pred_class, confidence, _ = predict_image(model, img_path)
            results.append({
                'filename': img_path.name,
                'prediction': pred_class,
                'confidence': confidence
            })
            print(f"✓ {img_path.name:30s} → {pred_class:10s} ({confidence:.1%})")
        except Exception as e:
            print(f"✗ {img_path.name:30s} → Error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH PREDICTION SUMMARY")
    print("=" * 60)
    for category in CLASSES:
        count = sum(1 for r in results if r['prediction'] == category)
        print(f"{category.upper():10s}: {count} items")
    print("=" * 60)


def main():
    """Main prediction interface"""
    
    print("=" * 60)
    print("WASTE SORTING MODEL - PREDICTION")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Single image:  python predict.py path/to/image.jpg")
        print("  Batch folder:  python predict.py path/to/folder/ --batch")
        print("\nExample:")
        print("  python predict.py test_images/bottle.jpg")
        print("  python predict.py test_images/ --batch")
        return
    
    image_path = sys.argv[1]
    
    # Check for batch mode
    if len(sys.argv) > 2 and sys.argv[2] == '--batch':
        predict_batch(image_path)
    elif Path(image_path).is_dir():
        predict_batch(image_path)
    else:
        predict_single_image(image_path)


if __name__ == "__main__":
    main()