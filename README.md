# Waste Sorting ML Project 🗑️♻️

A machine learning model that classifies waste into three categories: **Compost**, **Recycle**, and **Landfill**.

## 🚀 Quick Start

### 1. Setup Environment

First, create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Organize Your Dataset

The model expects this folder structure:

```
waste-sorting-project/
├── data/
│   ├── train/
│   │   ├── compost/     (put compost images here)
│   │   ├── recycle/     (put recycle images here)
│   │   └── landfill/    (put landfill images here)
│   └── val/
│       ├── compost/     (validation compost images)
│       ├── recycle/     (validation recycle images)
│       └── landfill/    (validation landfill images)
```

**Tips for collecting data:**
- Aim for at least 100-200 images per category
- Split about 80% for training, 20% for validation
- Include variety: different angles, lighting, backgrounds
- You can download datasets from:
  - [TrashNet](https://github.com/garythung/trashnet)
  - [TACO Dataset](http://tacodataset.org/)
  - Or take your own photos!

### 3. Train the Model

Once you have images in the folders:

```bash
python train.py
```

This will:
- Load and augment your images
- Build a CNN using transfer learning (MobileNetV2)
- Train for up to 20 epochs
- Save the best model to `models/waste_sorting_model.keras`
- Generate a training history plot

### 4. Make Predictions

Test your trained model on new images:

```bash
# Single image
python predict.py path/to/image.jpg

# All images in a folder
python predict.py path/to/folder/ --batch
```

## How This Project Works (Technical Deep Dive)

### The Big Picture

This waste sorting model uses **supervised learning** with labeled images. Instead of writing hard-coded rules like "if metal, then recycle," the model learns patterns from examples. After seeing thousands of images, it can classify new items it's never seen before.

Think of it like teaching a child to sort trash - you show examples and they learn what makes something recyclable vs. compost vs. landfill.

### Step-by-Step Process

#### 1. **Data Loading & Augmentation** (Lines 90-120 in train.py)

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixels from 0-255 to 0-1
    rotation_range=20,        # Randomly rotate images ±20°
    width_shift_range=0.2,    # Randomly shift horizontally
    height_shift_range=0.2,   # Randomly shift vertically
    horizontal_flip=True,     # Randomly flip images
    zoom_range=0.2           # Randomly zoom in/out
)
```

**Why this matters:**
- **Normalization**: Neural networks learn better with smaller numbers (0-1 instead of 0-255)
- **Augmentation**: Creates variations of each image so the model learns "what makes a banana a banana" rather than memorizing one specific photo. This prevents overfitting.
- Example: One banana photo becomes 10+ variations (rotated, flipped, zoomed) without collecting more data!

#### 2. **Transfer Learning - The Secret Sauce** (Lines 125-150)

```python
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze pre-trained layers
```

**This is the game-changer!**

Instead of training from scratch, we use **transfer learning**:
- MobileNetV2 was pre-trained on ImageNet (1.4 million images, 1000 categories)
- It already knows edges, textures, shapes, and basic object features
- We "freeze" these layers and only train the final classification layers
- Like learning Italian when you already know Spanish - you reuse existing knowledge!

**Why MobileNetV2?**
- Lightweight: ~3.4 million parameters (vs 25M+ in ResNet50)
- Fast: Trains quickly even on laptops
- Mobile-ready: Can deploy to phones/Raspberry Pi
- Accurate: Good balance of speed and performance

#### 3. **Custom Classification Head** (Lines 148-156)

```python
model = Sequential([
    base_model,                           # Feature extractor (frozen)
    GlobalAveragePooling2D(),             # Condense features
    Dropout(0.3),                         # Prevent overfitting
    Dense(128, activation='relu'),        # Learning layer
    Dropout(0.3),                         # More regularization
    Dense(3, activation='softmax')        # Final predictions
])
```

**Layer-by-layer breakdown:**

1. **base_model**: Extracts 1280 features from each image (edges, textures, patterns)
2. **GlobalAveragePooling2D**: Summarizes features into a single vector
3. **Dropout(0.3)**: Randomly "turns off" 30% of neurons during training (prevents memorization)
4. **Dense(128, relu)**: Learns combinations of features (128 neurons)
5. **Dropout(0.3)**: More regularization
6. **Dense(3, softmax)**: Outputs 3 probabilities that sum to 100%

**Softmax explained**: Converts raw scores to probabilities
- Example output: [0.75, 0.20, 0.05] = 75% compost, 20% recycle, 5% landfill
- The highest probability wins

#### 4. **The Learning Process** (Lines 159-163)

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Key components:**

- **Adam Optimizer**: Algorithm that adjusts model weights to improve predictions
  - Learning rate 0.001 = size of adjustments (too high = overshoots, too low = slow)
  - Like adjusting volume knobs - Adam figures out which knobs to turn and by how much

- **Categorical Cross-Entropy Loss**: Measures "wrongness" of predictions
  - Heavily penalizes confident wrong answers
  - Rewards confident correct answers
  - Example: Predicting 90% compost when it's landfill = huge penalty

- **Accuracy**: Simple metric we all understand (% correct predictions)

#### 5. **Training Loop** (Lines 197-206)

```python
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
)
```

**What happens each epoch (one complete pass through all data):**

1. Load batch of 32 images
2. Model makes predictions
3. Calculate loss (how wrong the predictions were)
4. **Backpropagation**: Adjust weights to reduce loss
5. Repeat for all batches
6. Test on validation data (never seen during training!)
7. Save if best performance so far
8. Repeat for up to 20 epochs

**Smart Callbacks (Safety Features):**

- **ModelCheckpoint**: Automatically saves the best version (highest validation accuracy)
- **EarlyStopping**: Stops if no improvement for 5 epochs (saves time, prevents overfitting)
- **ReduceLROnPlateau**: Lowers learning rate if stuck (like turning knobs more gently)

### Understanding Your Results

After training, you'll see a graph with two lines:

**Training vs. Validation Accuracy:**
- **Both increasing together** ✓ Great! Model is learning generalizable patterns
- **Training high, validation low** ✗ Overfitting (memorizing, not learning)
- **Both low** ✗ Underfitting (need more data or longer training)

**Example interpretation:**
"My model achieved 85% validation accuracy. This means when shown 100 images it's never seen before, it correctly classifies 85 of them. Training stopped at epoch 15 because accuracy plateaued - this prevents overfitting."

## Understanding the Output

### During Training
- **Accuracy**: How often the model predicts correctly (higher is better)
- **Loss**: How wrong the predictions are (lower is better)
- **Validation metrics**: Performance on unseen data (this is what really matters!)

### After Training
- Check `models/training_history.png` to see learning curves
- If training accuracy is much higher than validation accuracy → overfitting
- If both are low → need more data or longer training

## Improving Your Model

If accuracy is low, try:

1. **More data**: Collect more images (this usually helps most!)
2. **Better data quality**: Clear photos, correct labels
3. **More epochs**: Change `EPOCHS = 20` to `EPOCHS = 50` in train.py
4. **Data augmentation**: Already enabled (rotation, flipping, etc.)
5. **Different model**: Try ResNet50 instead of MobileNetV2

## How to Explain Your Project

### Simple Explanation (For Anyone)

"I built an AI that sorts waste by looking at pictures. Instead of writing rules, I showed it thousands of examples of compost, recyclables, and trash. The AI learned patterns - like shapes, colors, and textures - and can now classify new items it's never seen. It's like teaching a kid to sort trash, but the computer figures out the rules automatically."

### Technical Presentation Flow

1. **Problem Statement**
   - Manual waste sorting is error-prone and time-consuming
   - Contamination rates in recycling can reach 25%
   - Need for automated, accurate classification

2. **Solution Approach**
   - Supervised machine learning with Convolutional Neural Networks
   - Transfer learning from pre-trained MobileNetV2
   - Image classification into 3 categories: compost, recycle, landfill

3. **Dataset**
   - Collected [X] images across 3 categories
   - 80/20 train-validation split
   - Applied data augmentation (rotation, flipping, zooming)

4. **Model Architecture**
   - MobileNetV2 CNN with ImageNet pre-trained weights
   - Custom classification head with dropout regularization
   - Only ~140K trainable parameters (vs millions if training from scratch)

5. **Training Process**
   - Adam optimizer with learning rate 0.001
   - Categorical cross-entropy loss
   - Early stopping and learning rate scheduling
   - Trained for [X] epochs in [Y] minutes

6. **Results**
   - Achieved [X]% validation accuracy
   - Processes images in <1 second
   - Model size: ~14MB (deployable to mobile/edge devices)

7. **Future Improvements**
   - Deploy to Raspberry Pi with camera for real-time sorting
   - Multi-material detection (items with multiple components)
   - Integration with robotic sorting arms

### Key Technical Terms to Use

- **Supervised Learning**: Learning from labeled examples
- **Convolutional Neural Network (CNN)**: Neural network designed for image processing
- **Transfer Learning**: Using pre-trained model knowledge
- **Data Augmentation**: Creating variations to increase dataset size
- **Dropout Regularization**: Technique to prevent overfitting
- **Validation Accuracy**: Performance on unseen data
- **Softmax Activation**: Converts outputs to probabilities
- **Backpropagation**: Algorithm for updating model weights

### Common Questions & Answers

**Q: Why not just write if-else rules?**
A: "There are thousands of item types with infinite variations in appearance, lighting, angles, and condition. Writing rules for all cases is impossible. Machine learning automatically discovers patterns from data."

**Q: How much data do you need?**
A: "I used approximately 2,000 images total. Transfer learning allows success with less data - training from scratch would require 100,000+ images. The pre-trained model already knows general object features."

**Q: How long did training take?**
A: "15-20 minutes on a standard laptop CPU. With a GPU, it would take 3-5 minutes. Inference (making predictions) takes less than 1 second per image."

**Q: Can it handle items it's never seen before?**
A: "Yes! The model learns features (shapes, colors, textures, patterns) rather than memorizing specific images. However, accuracy decreases for items very different from the training data."

**Q: What are the limitations?**
A: "Challenges include: similar-looking items (paper vs cardboard), poor lighting conditions, multiple items in one image, and contaminated/dirty items. The model performs best on clear, single-item images similar to training data."

**Q: Why MobileNetV2 instead of other models?**
A: "MobileNetV2 offers the best balance of accuracy, speed, and efficiency. It's lightweight enough to run on mobile devices and Raspberry Pi, making deployment practical. Larger models like ResNet would be more accurate but slower and harder to deploy."

**Q: How do you prevent overfitting?**
A: "Multiple techniques: data augmentation creates more training variations, dropout randomly disables neurons during training, validation data monitors generalization, and early stopping prevents excessive training. Transfer learning also helps by starting with general knowledge."

### Metrics to Highlight

- **Accuracy**: Primary metric (% of correct predictions)
- **Training Time**: Shows practical feasibility
- **Inference Speed**: Important for real-time applications
- **Model Size**: Relevant for deployment constraints
- **Precision/Recall per Category**: Shows per-class performance

### Demo Ideas

1. **Live Classification**: Show the model classifying items in real-time
2. **Confidence Scores**: Display the probability distribution for each prediction
3. **Failure Cases**: Show examples where it struggles (builds credibility)
4. **Augmentation Examples**: Show how the same image is varied during training
5. **Training Curves**: Display the learning progress graph

## Project Structure

```
waste-sorting-project/
├── train.py              # Training script
├── predict.py            # Prediction script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Your dataset (you create this)
└── models/              # Saved models and plots
```

## VS Code Tips

1. **Install Python extension**: Search "Python" in extensions
2. **Select interpreter**: Ctrl+Shift+P → "Python: Select Interpreter" → Choose your venv
3. **Run scripts**: Right-click on train.py → "Run Python File in Terminal"
4. **Debugging**: Set breakpoints by clicking left of line numbers

## Next Steps

1. Set up environment
2. Collect and organize dataset
3. Train your first model
4. Evaluate performance
5. Iterate and improve
6. Deploy (future step!)

## Getting Your Dataset

### Option 1: Kaggle Datasets (Recommended)

**Garbage Classification Dataset**
- Has biological/organic waste (your compost category!)
- 12 categories total
- Link: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

**Steps:**
1. Create free Kaggle account
2. Download dataset ZIP
3. Extract to a folder
4. Use `organize_dataset.py` script to sort into train/val folders

**Category Mapping:**
- Compost: biological
- Recycle: cardboard, paper, plastic, metal, glass (all types)
- Landfill: trash, battery, clothes, shoes

### Option 2: TrashNet (GitHub)

```bash
git clone https://github.com/garythung/trashnet.git
```

- 2,527 images in 6 categories
- No organic/compost category (you'll need to add your own)
- Good for recycle/landfill categories

### Option 3: Create Your Own Dataset

**Most authentic approach:**
1. Take 100-200 photos per category with your phone
2. Use various angles, lighting, backgrounds
3. Include items you actually throw away
4. Benefits: Model works perfectly for YOUR specific use case

**Photography tips:**
- Clear, well-lit photos
- Single item per photo (or label accordingly)
- Include variations: clean, dirty, crushed, whole
- Different backgrounds to improve generalization

### organize_dataset.py Script

Save this to organize downloaded datasets:

```python
# organize_dataset.py
import os
import shutil
from pathlib import Path
import random

# UPDATE THIS PATH to where you extracted the dataset
DATASET_PATH = "path/to/your/downloaded/dataset"

PROJECT_DATA = "data"

# Map source categories to your 3 target categories
CATEGORY_MAP = {
    'compost': ['biological'],
    'recycle': ['cardboard', 'paper', 'plastic', 'metal', 
                'brown-glass', 'green-glass', 'white-glass'],
    'landfill': ['trash', 'battery', 'clothes', 'shoes']
}

TRAIN_RATIO = 0.8

def organize_images():
    print("Starting dataset organization...\n")
    
    for target_category, source_categories in CATEGORY_MAP.items():
        all_images = []
        print(f"Processing {target_category.upper()}:")
        
        for source_cat in source_categories:
            source_path = Path(DATASET_PATH) / source_cat
            if source_path.exists():
                images = (list(source_path.glob('*.jpg')) + 
                         list(source_path.glob('*.jpeg')) + 
                         list(source_path.glob('*.png')))
                all_images.extend(images)
                print(f"  ✓ {source_cat}: {len(images)} images")
            else:
                print(f"  ✗ {source_cat}: folder not found")
        
        if not all_images:
            print(f"  ⚠ No images found for {target_category}\n")
            continue
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * TRAIN_RATIO)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        # Create directories and copy
        train_dir = Path(PROJECT_DATA) / 'train' / target_category
        val_dir = Path(PROJECT_DATA) / 'val' / target_category
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(train_images):
            shutil.copy(img, train_dir / f"{target_category}_{i:04d}{img.suffix}")
        
        for i, img in enumerate(val_images):
            shutil.copy(img, val_dir / f"{target_category}_val_{i:04d}{img.suffix}")
        
        print(f"  ✓ {len(train_images)} train, {len(val_images)} val\n")

if __name__ == "__main__":
    organize_images()
    print("✓ Dataset organized! Run: python train.py")
```

## Troubleshooting

**"No module named tensorflow"**
- Make sure your virtual environment is activated
- Run `pip install -r requirements.txt` again

**"No training data found"**
- Check that images are in `data/train/compost/`, etc.
- Run `python train.py` first to create folders

**Low accuracy**
- Need more training images (100+ per category minimum)
- Make sure images are clearly labeled in correct folders
- Train for more epochs

**Out of memory**
- Reduce `BATCH_SIZE` in train.py (try 16 or 8)

**"Could not load dynamic library 'cudart64_110.dll'"** (Windows GPU warning)
- This is just a warning - TensorFlow will use CPU instead
- Model still works, just slower
- Ignore unless you have a GPU and want to use it

**Training is very slow**
- Normal on CPU (15-20 minutes)
- Consider Google Colab for free GPU access
- Or reduce image count for testing

**Validation accuracy not improving**
- Try more epochs (change `EPOCHS = 20` to `EPOCHS = 50`)
- Check if images are correctly labeled
- May need more diverse training data

**"ResourceExhaustedError" / "OOM" (Out of Memory)**
- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Reduce `IMG_SIZE` from 224 to 160 (less accurate but uses less RAM)
- Close other applications

**Model predicts same class for everything**
- Check class balance (equal images per category)
- Verify images are in correct folders
- Try training longer (more epochs)

## Learning Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Image Classification Best Practices](https://www.tensorflow.org/tutorials/images/classification)
- [Understanding CNNs](https://poloclub.github.io/cnn-explainer/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

## Advanced Topics

### Fine-Tuning the Base Model

After initial training, you can "unfreeze" some layers of MobileNetV2 for better accuracy:

```python
# In train.py, after initial training
base_model.trainable = True

# Freeze early layers, train only last 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train a few more epochs
model.fit(train_gen, epochs=10, validation_data=val_gen)
```

### Confusion Matrix Analysis

See which categories the model confuses:

```python
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Get predictions
val_generator.reset()
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

# Print confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
print(classification_report(y_true, y_pred, target_names=CLASSES))
```

### Grad-CAM Visualization

See what the model is "looking at":

```python
# Shows which parts of the image influenced the decision
# Useful for debugging and building trust
# Implementation available in TensorFlow tutorials
```

### Deployment Options

**Option 1: Flask Web App**
```python
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/waste_sorting_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    # ... process and predict
    return jsonify({'category': prediction})
```

**Option 2: Raspberry Pi + Camera**
- Run inference on edge device
- Real-time waste sorting at the bin
- Can trigger physical sorting mechanisms

**Option 3: Mobile App**
- Convert to TensorFlow Lite (.tflite)
- Reduces model size by ~75%
- Runs on Android/iOS

### Model Optimization

**TensorFlow Lite Conversion:**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Quantization** (makes model 4x smaller):
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

### Multi-Label Classification

If items can be multiple categories (e.g., dirty paper = compost AND paper):

```python
# Change loss function
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Instead of categorical
    metrics=['accuracy']
)
```

### Data Balancing

If you have uneven categories:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

# Use in training
model.fit(train_gen, class_weight=dict(enumerate(class_weights)))
```

## Project Extensions

1. **Real-time webcam classification**
2. **Barcode scanning for packaging info**
3. **Multi-language support for labels**
4. **Carbon footprint calculator per category**
5. **Gamification (points for correct sorting)**
6. **Community data collection app**
7. **Integration with smart bins**
8. **Recycling facility recommendation**

Good luck with your project! 🎉
