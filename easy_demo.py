"""
EASY DEMO - Sanskrit OCR System
Simple script to process Sanskrit manuscript images

How to use:
1. Put your Sanskrit manuscript images in a folder
2. Change the FOLDER_PATH below to point to your images
3. Run this script: python easy_demo.py
4. Results will be saved in the output folder
"""

from model2 import SanskritOCR
import os

def demo():
    print("=== Sanskrit OCR Demo ===")
    print("This will process all images in your specified folder")
    
    # ============================================
    # CHANGE THESE PATHS TO YOUR FOLDERS
    # ============================================
    
    # Path to folder containing your Sanskrit images
    FOLDER_PATH = r"d:\Projects\Sanskrit-OCR\maitrayani_dataset\line_images"
    
    # Path where results will be saved
    OUTPUT_PATH = r"d:\Projects\Sanskrit-OCR\demo_results"
    
    # ============================================
    
    # Check if folder exists
    if not os.path.exists(FOLDER_PATH):
        print(f"ERROR: Folder not found: {FOLDER_PATH}")
        print("Please update FOLDER_PATH to point to your images folder")
        return
    
    print(f"Input folder: {FOLDER_PATH}")
    print(f"Output folder: {OUTPUT_PATH}")
    
    # Initialize the OCR system
    print("\nInitializing Sanskrit OCR system...")
    ocr = SanskritOCR()
    
    # Process all images
    print(f"\nStarting to process images...")
    results = ocr.process_folder(FOLDER_PATH, OUTPUT_PATH)
    
    # Show summary
    total = len(results)
    successful = sum(1 for r in results if r['processing_status'] == 'success')
    failed = total - successful
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*50}")
    print(f"Total images: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\nResults saved to: {OUTPUT_PATH}")
        print("Files created:")
        print("- all_results.json (complete results)")
        print("- results_summary.csv (summary table)")
        print("- individual result files for each image")
        
        # Show first successful result as example
        for result in results:
            if result['processing_status'] == 'success':
                image_name = os.path.basename(result['image_path'])
                print(f"\nSample result from {image_name}:")
                print(f"Sanskrit text: {result['sanskrit_text'][:100]}...")
                print(f"Transliterated: {result['transliterated_text'][:100]}...")
                break
    
    if failed > 0:
        print(f"\nFailed images:")
        for result in results:
            if result['processing_status'] == 'failed':
                print(f"- {os.path.basename(result['image_path'])}: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    demo()
