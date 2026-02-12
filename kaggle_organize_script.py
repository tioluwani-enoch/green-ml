# organize_dataset.py
import os
import shutil
from pathlib import Path
import random

# Where your dataset folders are (the folder containing battery, biological, etc.)
DATASET_PATH = "kaggle_dataset/garbage_classification"  # UPDATE THIS!

# Your project data folder
PROJECT_DATA = "data"

# Mapping dataset categories to your 3 categories
CATEGORY_MAP = {
    'compost': ['biological'],  # ✓ This is your organic waste!
    'recycle': ['cardboard', 'paper', 'plastic', 'metal', 'brown-glass', 'green-glass', 'white-glass'],
    'landfill': ['trash', 'battery', 'clothes', 'shoes']
}

# Split ratio (80% train, 20% validation)
TRAIN_RATIO = 0.8

def organize_images():
    """Organize dataset images into train/val splits"""
    
    print("Starting dataset organization...\n")
    
    for target_category, source_categories in CATEGORY_MAP.items():
        all_images = []
        
        print(f"Processing {target_category.upper()}:")
        
        # Collect all images for this category
        for source_cat in source_categories:
            source_path = Path(DATASET_PATH) / source_cat
            if source_path.exists():
                # Get all image files
                images = (list(source_path.glob('*.jpg')) + 
                         list(source_path.glob('*.jpeg')) + 
                         list(source_path.glob('*.png')))
                all_images.extend(images)
                print(f"  ✓ {source_cat}: {len(images)} images")
            else:
                print(f"  ✗ {source_cat}: folder not found at {source_path}")
        
        if not all_images:
            print(f"  ⚠ WARNING: No images found for {target_category}!\n")
            continue
        
        # Shuffle and split into train/validation
        random.seed(42)  # For reproducibility
        random.shuffle(all_images)
        split_idx = int(len(all_images) * TRAIN_RATIO)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        # Create directories
        train_dir = Path(PROJECT_DATA) / 'train' / target_category
        val_dir = Path(PROJECT_DATA) / 'val' / target_category
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy to train folder
        print(f"  Copying {len(train_images)} images to train/{target_category}/...")
        for i, img in enumerate(train_images):
            # Rename to avoid conflicts
            new_name = f"{target_category}_{i:04d}{img.suffix}"
            shutil.copy(img, train_dir / new_name)
        
        # Copy to val folder
        print(f"  Copying {len(val_images)} images to val/{target_category}/...")
        for i, img in enumerate(val_images):
            new_name = f"{target_category}_val_{i:04d}{img.suffix}"
            shutil.copy(img, val_dir / new_name)
        
        print(f"  ✓ TOTAL: {len(train_images)} train, {len(val_images)} val\n")

def print_summary():
    """Print summary of organized dataset"""
    print("="*60)
    print("DATASET ORGANIZATION COMPLETE!")
    print("="*60)
    
    categories = ['compost', 'recycle', 'landfill']
    
    for category in categories:
        train_path = Path(PROJECT_DATA) / 'train' / category
        val_path = Path(PROJECT_DATA) / 'val' / category
        
        train_count = len(list(train_path.glob('*'))) if train_path.exists() else 0
        val_count = len(list(val_path.glob('*'))) if val_path.exists() else 0
        
        print(f"{category.upper():10s}: {train_count:4d} train | {val_count:4d} val | {train_count + val_count:4d} total")
    
    print("="*60)
    print("\n✓ Ready to train! Run: python train.py")

if __name__ == "__main__":
    # Check if dataset path exists
    if not Path(DATASET_PATH).exists():
        print("❌ ERROR: Dataset path not found!")
        print(f"Please update DATASET_PATH in the script to point to your dataset folder")
        print(f"Current path: {DATASET_PATH}")
        exit(1)
    
    organize_images()
    print_summary()