"""
🕉️ SANSKRIT OCR DEMO SCRIPT
============================

This script demonstrates how to use the Sanskrit OCR system
with your existing dataset. It provides simple examples
that you can run immediately.

Usage:
    python demo.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def check_dataset():
    """
    Check if the dataset exists and show information about it
    """
    print("🔍 CHECKING DATASET STATUS")
    print("=" * 30)
    
    dataset_dir = Path("maitrayani_dataset")
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found!")
        print("💡 Please run 'python create_dataset.py' first to create the dataset.")
        return None
    
    # Check for dataset files
    final_dataset_dir = dataset_dir / "final_dataset"
    aligned_csv = final_dataset_dir / "aligned_dataset.csv"
    
    if not aligned_csv.exists():
        print("❌ Dataset CSV file not found!")
        print("💡 Please run 'python create_dataset.py' to generate the complete dataset.")
        return None
    
    # Load and show dataset info
    try:
        df = pd.read_csv(aligned_csv)
        
        print("✅ Dataset found and loaded successfully!")
        print(f"📊 Total records: {len(df)}")
        print(f"📁 Pages: {df['page_number'].nunique()}")
        print(f"📝 Records with text: {df['has_ground_truth'].sum()}")
        print(f"🖼️  Records with images: {df['image_exists'].sum()}")
        
        # Check if training splits exist
        splits = ['train_split.csv', 'validation_split.csv', 'test_split.csv']
        split_info = {}
        
        for split_file in splits:
            split_path = final_dataset_dir / split_file
            if split_path.exists():
                split_df = pd.read_csv(split_path)
                split_name = split_file.replace('_split.csv', '')
                split_info[split_name] = len(split_df)
                print(f"🏋️ {split_name.title()} split: {len(split_df)} records")
        
        return {
            'dataset_path': str(aligned_csv),
            'total_records': len(df),
            'dataset_dir': str(dataset_dir),
            'splits': split_info
        }
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def demo_single_image():
    """
    Demo: Process a single image from your dataset
    """
    print("\n🖼️ SINGLE IMAGE PROCESSING DEMO")
    print("=" * 35)
    
    # Check dataset first
    dataset_info = check_dataset()
    if not dataset_info:
        return
    
    # Load dataset to find a good example
    df = pd.read_csv(dataset_info['dataset_path'])
    
    # Find records that have both image and text
    good_records = df[(df['has_ground_truth'] == True) & (df['image_exists'] == True)]
    
    if len(good_records) == 0:
        print("❌ No records found with both image and text!")
        return
    
    # Pick the first good record
    sample_record = good_records.iloc[0]
    
    print(f"📸 Sample image: {sample_record['image_path']}")
    print(f"📝 Ground truth: {sample_record['ground_truth_text'][:100]}...")
    
    # Check if image file exists
    image_path = sample_record['image_path']
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        print("💡 Make sure the line_images directory contains the segmented images")
        return
    
    print("✅ Image file exists!")
    print("\n💡 TO PROCESS THIS IMAGE WITH OCR:")
    print("Run the following command:")
    print(f"python -c \"")
    print(f"from model import SanskritOCR")
    print(f"ocr = SanskritOCR()")
    print(f"result = ocr.process_single_image('{image_path}', '{sample_record['ground_truth_text']}')")
    print(f"print('Results:', result)")
    print(f"\"")

def demo_dataset_info():
    """
    Demo: Show detailed information about your dataset
    """
    print("\n📊 DATASET INFORMATION DEMO")
    print("=" * 30)
    
    dataset_info = check_dataset()
    if not dataset_info:
        return
    
    # Load full dataset
    df = pd.read_csv(dataset_info['dataset_path'])
    
    # Show detailed statistics
    print("\n📈 DETAILED STATISTICS:")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique pages: {df['page_number'].nunique()}")
    print(f"   Average lines per page: {df.groupby('page_number').size().mean():.1f}")
    
    # Text statistics
    text_records = df[df['has_ground_truth'] == True]
    if len(text_records) > 0:
        print(f"\n📝 TEXT STATISTICS:")
        print(f"   Records with text: {len(text_records)}")
        print(f"   Average text length: {text_records['text_length'].mean():.1f} characters")
        print(f"   Shortest text: {text_records['text_length'].min()} characters")
        print(f"   Longest text: {text_records['text_length'].max()} characters")
    
    # Image statistics
    image_records = df[df['image_exists'] == True]
    print(f"\n🖼️  IMAGE STATISTICS:")
    print(f"   Records with images: {len(image_records)}")
    print(f"   Image coverage: {(len(image_records) / len(df)) * 100:.1f}%")
    
    # Usable for training
    training_ready = df[(df['has_ground_truth'] == True) & (df['image_exists'] == True)]
    print(f"\n🏋️ TRAINING READINESS:")
    print(f"   Records ready for training: {len(training_ready)}")
    print(f"   Training coverage: {(len(training_ready) / len(df)) * 100:.1f}%")
    
    # Show sample data
    print(f"\n📋 SAMPLE RECORDS:")
    sample_records = training_ready.head(3)
    for idx, record in sample_records.iterrows():
        print(f"   Record {idx + 1}:")
        print(f"     📄 Page {record['page_number']}, Line {record['line_number']}")
        print(f"     📝 Text: {record['ground_truth_text'][:50]}...")
        print(f"     🖼️  Image: {os.path.basename(record['image_path'])}")

def demo_ocr_usage():
    """
    Show how to use the OCR system with your dataset
    """
    print("\n🚀 OCR USAGE EXAMPLES")
    print("=" * 25)
    
    print("Here are example commands to use your Sanskrit OCR system:\n")
    
    print("1️⃣ PROCESS A SINGLE IMAGE:")
    print("```python")
    print("from model import SanskritOCR")
    print("ocr = SanskritOCR()")
    print("result = ocr.process_single_image('path/to/your/image.png')")
    print("print(result['final_output'])")
    print("```\n")
    
    print("2️⃣ PROCESS YOUR ENTIRE DATASET:")
    print("```python")
    print("from model import SanskritOCR")
    print("ocr = SanskritOCR()")
    print("summary = ocr.process_dataset('maitrayani_dataset/final_dataset/aligned_dataset.csv')")
    print("print(f'Processed {summary[\"successful_processes\"]} images successfully!')")
    print("```\n")
    
    print("3️⃣ USE TRAINING SPLITS FOR ML:")
    print("```python")
    print("import pandas as pd")
    print("train_df = pd.read_csv('maitrayani_dataset/final_dataset/train_split.csv')")
    print("val_df = pd.read_csv('maitrayani_dataset/final_dataset/validation_split.csv')")
    print("test_df = pd.read_csv('maitrayani_dataset/final_dataset/test_split.csv')")
    print("# Use these for training your own OCR model")
    print("```\n")
    
    print("4️⃣ INTERACTIVE MODE:")
    print("```bash")
    print("python model.py")
    print("# Follow the interactive prompts")
    print("```\n")

def main():
    """
    Main demo function
    """
    print("🕉️ SANSKRIT OCR DEMO")
    print("=" * 20)
    print("Welcome! This demo will show you how to use your Sanskrit OCR dataset.\n")
    
    # Check dataset status
    dataset_info = check_dataset()
    
    if dataset_info:
        print("\n🎯 RUNNING DEMOS:")
        
        # Show dataset information
        demo_dataset_info()
        
        # Show single image demo
        demo_single_image()
        
        # Show usage examples
        demo_ocr_usage()
        
        print("\n🎉 DEMO COMPLETED!")
        print("=" * 20)
        print("Your dataset is ready to use!")
        print("You can now:")
        print("✅ Process individual images with the OCR system")
        print("✅ Train your own OCR models using the training splits")
        print("✅ Evaluate OCR performance using the test set")
        print("✅ Use the transliteration and accent detection features")
        
    else:
        print("\n💡 NEXT STEPS:")
        print("1. Run 'python create_dataset.py' to create your dataset")
        print("2. Run this demo again to see your dataset information")
        print("3. Use 'python model.py' to start processing images")

if __name__ == "__main__":
    main()
