"""
Waste Sorting - Phone Camera Web App
A simple web interface that works on your phone or computer browser
Run this and open http://localhost:5000 on your device
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configuration
MODEL_PATH = "models/waste_sorting_model.keras"
IMG_SIZE = 224
CLASSES = ['compost', 'recycle', 'landfill']

# Load model once at startup
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("✓ Model loaded!")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess
        image = image.convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get results
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Return all predictions
        result = {
            'category': predicted_class,
            'confidence': confidence,
            'all_predictions': {
                class_name: float(pred) 
                for class_name, pred in zip(CLASSES, predictions)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("WASTE SORTING WEB APP")
    print("=" * 60)
    print("\nOpen in your browser:")
    print("  Computer: http://localhost:5000")
    print("  Phone: http://YOUR_COMPUTER_IP:5000")
    print("\nTo find your computer's IP:")
    print("  Windows: ipconfig")
    print("  Mac/Linux: ifconfig")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)