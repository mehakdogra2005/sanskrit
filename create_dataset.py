"""
Complete Dataset Creation Script for Maitrayani Samhita OCR
This script creates the complete dataset as described in the research paper:

- Downloads TITUS electronic text with page/line numbers
- Processes PDF to extract page images 
- Performs line segmentation using haar-like features
- Creates aligned image-text pairs
- Generates training/validation/test splits
- Produces final dataset with 11,740+ line images and corresponding text

Usage:
    python create_dataset.py

Output:
    - maitrayani_dataset/final_dataset/ : Complete aligned dataset
    - Training, validation, and test splits
    - Comprehensive statistics and analysis
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our dataset creator
from datacollect import MaitrayaniDatasetCreator

def install_requirements():
    """Install required packages"""
    import subprocess
    
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        print("Please install manually: pip install -r requirements.txt")
        return False

def create_complete_dataset():
    """Create the complete Maitrayani Samhita OCR dataset"""
    
    print("=" * 60)
    print("MAITRAYANI SAMHITA OCR DATASET CREATION")
    print("=" * 60)
    print()
    print("This script will:")
    print("1. Download TITUS electronic text with page/line markers")
    print("2. Download PDF from Archive.org") 
    print("3. Convert PDF pages to high-resolution images")
    print("4. Segment text lines using haar-like features")
    print("5. Align line images with corresponding text")
    print("6. Create training/validation/test splits")
    print("7. Generate comprehensive statistics")
    print()
    
    # Ask for user confirmation
    response = input("Proceed with dataset creation? (y/n): ")
    if response.lower() != 'y':
        print("Dataset creation cancelled.")
        return
    
    print("\nInitializing dataset creator...")
    
    # Create dataset creator
    creator = MaitrayaniDatasetCreator("maitrayani_dataset")
    
    try:
        # Create the complete dataset
        print("\nStarting dataset creation process...")
        dataset_path = creator.create_final_dataset()
        
        if dataset_path:
            print("\n" + "=" * 60)
            print("DATASET CREATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"\nDataset location: {dataset_path}")
            print("\nGenerated files:")
            print("📊 aligned_dataset.csv - Complete image-text alignment")
            print("🏋️ train_split.csv - Training data")
            print("✅ validation_split.csv - Validation data")
            print("🧪 test_split.csv - Test data")
            print("📈 dataset_statistics.json - Comprehensive statistics")
            print("📋 dataset_summary.json - Overall summary")
            
            print("\nDirectory structure:")
            print("📁 page_images/ - Original PDF pages as images")
            print("📁 line_images/ - Segmented text line images") 
            print("📁 text_data/ - TITUS electronic text extraction")
            print("📁 final_dataset/ - Complete aligned dataset")
            
            # Load and display final statistics
            try:
                import pandas as pd
                import json
                
                # Load dataset
                df = pd.read_csv(os.path.join(dataset_path, "aligned_dataset.csv"))
                
                # Load statistics
                with open(os.path.join(dataset_path, "dataset_statistics.json"), 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                print("\n📊 FINAL DATASET STATISTICS:")
                print(f"   Total records: {len(df):,}")
                print(f"   Pages processed: {df['page_number'].nunique()}")
                print(f"   Average lines per page: {df.groupby('page_number').size().mean():.1f}")
                print(f"   Records with ground truth text: {df['has_ground_truth'].sum():,}")
                print(f"   Text coverage: {(df['has_ground_truth'].sum() / len(df)) * 100:.1f}%")
                print(f"   Images available: {df['image_exists'].sum():,}")
                print(f"   Image coverage: {(df['image_exists'].sum() / len(df)) * 100:.1f}%")
                
                # Training split info
                usable_records = df[(df['has_ground_truth']) & (df['image_exists'])]
                print(f"   Usable for training: {len(usable_records):,}")
                
                print("\n🎯 Ready for OCR model training!")
                print("\nNext steps:")
                print("1. Use train_split.csv for model training")
                print("2. Use validation_split.csv for hyperparameter tuning")
                print("3. Use test_split.csv for final evaluation")
                print("4. Line images are in line_images/ directory")
                print("5. Ground truth text is in the 'ground_truth_text' column")
                
            except Exception as e:
                print(f"\nNote: Could not load statistics for display: {e}")
                print("But dataset files have been created successfully!")
                
        else:
            print("\n❌ Dataset creation failed!")
            print("Please check the error messages above and try again.")
            
    except Exception as e:
        print(f"\n❌ Error during dataset creation: {e}")
        print("Please check the error details above.")
        return

def main():
    """Main function"""
    
    print("Maitrayani Samhita OCR Dataset Creator")
    print("=====================================\n")
    
    # Check if requirements are installed
    try:
        import requests
        import cv2
        import pandas as pd
        from bs4 import BeautifulSoup
        import fitz
        from PIL import Image
        print("✓ All required packages are available")
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("\nAttempting to install requirements...")
        
        if not install_requirements():
            print("Please install requirements manually and try again.")
            return
    
    # Create the dataset
    create_complete_dataset()

if __name__ == "__main__":
    main()
