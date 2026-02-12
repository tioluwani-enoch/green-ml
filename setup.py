"""
Quick Setup Script
Run this first to create the folder structure and verify everything is ready
"""

import os
import sys

def create_directories():
    """Create all necessary directories"""
    directories = [
        "data/train/compost",
        "data/train/recycle",
        "data/train/landfill",
        "data/val/compost",
        "data/val/recycle",
        "data/val/landfill",
        "models",
        "test_images"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    print("\n✓ All directories created!")


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'tensorflow',
        'numpy',
        'matplotlib',
        'PIL',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠ Warning: Some packages are missing!")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def count_images():
    """Count images in each directory"""
    print("\nCounting images in dataset...")
    
    categories = ['compost', 'recycle', 'landfill']
    splits = ['train', 'val']
    
    total_train = 0
    total_val = 0
    
    print("\nTRAINING DATA:")
    for category in categories:
        path = f"data/train/{category}"
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            total_train += count
            print(f"  {category:10s}: {count:3d} images")
    
    print(f"\n  Total: {total_train} training images")
    
    print("\nVALIDATION DATA:")
    for category in categories:
        path = f"data/val/{category}"
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            total_val += count
            print(f"  {category:10s}: {count:3d} images")
    
    print(f"\n  Total: {total_val} validation images")
    
    if total_train == 0:
        print("\n⚠ No training images found!")
        print("Please add images to data/train/ folders before training.")
        return False
    elif total_train < 50:
        print("\n⚠ Very few training images!")
        print("Recommendation: At least 100-200 images per category for good results.")
        return True
    else:
        print("\n✓ Good amount of data!")
        return True


def main():
    print("=" * 60)
    print("WASTE SORTING PROJECT - SETUP")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Count images
    data_ok = count_images()
    
    # Final summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if deps_ok and data_ok:
        print("✓ Everything looks good!")
        print("\nNext steps:")
        print("  1. Make sure you have enough images in data/train/ folders")
        print("  2. Run: python train.py")
    elif not deps_ok:
        print("⚠ Missing dependencies")
        print("\nNext step:")
        print("  1. Run: pip install -r requirements.txt")
    else:
        print("⚠ Need to add training data")
        print("\nNext steps:")
        print("  1. Add images to data/train/compost/, recycle/, landfill/")
        print("  2. Add images to data/val/compost/, recycle/, landfill/")
        print("  3. Run: python train.py")
    
    print("=" * 60)


if __name__ == "__main__":
    main()