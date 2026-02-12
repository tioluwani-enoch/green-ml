# Waste Sorting ML Project 🗑️♻️

A machine learning model that classifies waste into three categories: **Compost**, **Recycle**, and **Landfill**.

## Quick Start

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
green-ml/
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

## Understanding the Output

### During Training
- **Accuracy**: How often the model predicts correctly (higher is better)
- **Loss**: How wrong the predictions are (lower is better)
- **Validation metrics**: Performance on unseen data (this is what really matters!)

### After Training
- Check `models/training_history.png` to see learning curves
- If training accuracy is much higher than validation accuracy → overfitting
- If both are low → need more data or longer training

## 🛠️ Improving Your Model

If accuracy is low, try:

1. **More data**: Collect more images (this usually helps most!)
2. **Better data quality**: Clear photos, correct labels
3. **More epochs**: Change `EPOCHS = 20` to `EPOCHS = 50` in train.py
4. **Data augmentation**: Already enabled (rotation, flipping, etc.)
5. **Different model**: Try ResNet50 instead of MobileNetV2

## Project Structure
```
green-ml/
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

## Learning Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Image Classification Best Practices](https://www.tensorflow.org/tutorials/images/classification)

Good luck with your project!