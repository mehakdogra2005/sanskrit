import cv2
import numpy as np
import os
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from typing import List, Tuple
import re
import json
import pandas as pd
from datetime import datetime

class SanskritOCR:
    def __init__(self):
        """Initialize the Sanskrit OCR system"""
        print("Initializing Sanskrit OCR...")
        
        # Load Microsoft TrOCR model for text recognition
        try:
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load TrOCR model: {e}")
            print("Falling back to Tesseract OCR")
            self.processor = None
            self.model = None
    
    def preprocess_image_for_ocr(self, image_path: str) -> np.ndarray:
        """
        Clean up the image so the computer can read it better
        Steps: Convert to grayscale -> Remove noise -> Binarize -> Line segmentation
        """
        print(f"Preprocessing image: {os.path.basename(image_path)}")
        
        # Step 1: Convert colored image to black and white (grayscale)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Remove noise/dots/scratches from old paper (denoising)
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Step 3: Make text pure black and background pure white (binarization)
        binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        print("Image preprocessing completed")
        return binary
    
    def segment_lines(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Separate each line of text from the full page
        Like using a ruler to separate each line of handwritten text into individual strips
        """
        print("Segmenting lines from the image...")
        
        # Horizontal Projection: Count black pixels in each row
        h_projection = np.sum(binary_image == 0, axis=1)
        
        # Find boundaries: Detect where text lines start and end
        lines = []
        in_line = False
        start_y = 0
        
        for i, pixel_count in enumerate(h_projection):
            if pixel_count > 10 and not in_line:  # Start of a line
                start_y = i
                in_line = True
            elif pixel_count <= 10 and in_line:  # End of a line
                if i - start_y > 20:  # Filter: Remove lines that are too small
                    line_image = binary_image[start_y:i, :]
                    lines.append(line_image)
                in_line = False
        
        print(f"Found {len(lines)} text lines")
        return lines
    
    def recognize_text(self, line_image: np.ndarray) -> str:
        """
        Convert image of text into actual computer text
        Like having a very smart robot that can look at a picture and tell you what Sanskrit words are written in it
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(line_image)
            
            # Use TrOCR model if available
            if self.model is not None and self.processor is not None:
                # Technical Process:
                # 1. Processor(image) → Converts image to numbers the AI can process
                # 2. Model.generate() → AI brain processes the numbers and makes predictions  
                # 3. Processor.decode() → Converts AI's predictions back to Sanskrit text
                
                pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text
            else:
                # Fallback to Tesseract with Sanskrit configuration
                custom_config = r'-l san --oem 3 --psm 6'
                text = pytesseract.image_to_string(pil_image, config=custom_config)
                return text.strip()
                
        except Exception as e:
            print(f"Error in text recognition: {e}")
            return ""
    
    def correct_text(self, recognized_text: str) -> str:
        """
        Fix mistakes in the recognized text
        Like a spell-checker but specifically for Sanskrit that knows which mistakes are more serious
        """
        if not recognized_text:
            return recognized_text
        
        # Basic corrections for common OCR errors
        corrections = {
            'rn': 'm',  # Common OCR mistake
            'ii': 'ी',  # Devanagari long i
            'aa': 'ा',  # Devanagari long a
            '0': 'ो',   # Zero often confused with Devanagari o
            '1': 'l',   # One often confused with l
        }
        
        corrected_text = recognized_text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)
        
        return corrected_text
    
    def transliterate_to_iso15919(self, sanskrit_text: str) -> str:
        """
        Convert Sanskrit script to English letters using international rules
        Maps each Sanskrit character to English equivalent while preserving pronunciation
        """
        # Basic transliteration mapping (simplified version)
        mapping = {
            'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū',
            'ए': 'e', 'ओ': 'o', 'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha',
            'च': 'ca', 'छ': 'cha', 'ज': 'ja', 'झ': 'jha', 'ट': 'ṭa', 'ठ': 'ṭha',
            'ड': 'ḍa', 'ढ': 'ḍha', 'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha',
            'न': 'na', 'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
            'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'śa', 'ष': 'ṣa',
            'स': 'sa', 'ह': 'ha', 'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū',
            'े': 'e', 'ो': 'o', '्': '', 'ं': 'ṃ', 'ः': 'ḥ'
        }
        
        transliterated = ""
        for char in sanskrit_text:
            transliterated += mapping.get(char, char)
        
        return transliterated
    
    def detect_accent_marks(self, sanskrit_text: str) -> dict:
        """
        Find and count special pronunciation marks in Vedic Sanskrit
        Udatta: High tone (not usually marked)
        Anudatta: Low tone (not usually marked)
        """
        accent_info = {
            'udatta_count': 0,
            'anudatta_count': 0,
            'total_accents': 0,
            'has_vedic_accents': False
        }
        
        # Count various accent marks (if present in the text)
        udatta_marks = ['॑', '᳭']  # Vedic tone marks
        anudatta_marks = ['॒', '᳜']
        
        for mark in udatta_marks:
            accent_info['udatta_count'] += sanskrit_text.count(mark)
        
        for mark in anudatta_marks:
            accent_info['anudatta_count'] += sanskrit_text.count(mark)
        
        accent_info['total_accents'] = accent_info['udatta_count'] + accent_info['anudatta_count']
        accent_info['has_vedic_accents'] = accent_info['total_accents'] > 0
        
        return accent_info
    
    def process_single_image(self, image_path: str) -> dict:
        """Process a single image and return results"""
        try:
            # Step 1: PREPROCESSING (Getting images ready)
            binary_image = self.preprocess_image_for_ocr(image_path)
            
            # Step 2: Line Segmentation
            lines = self.segment_lines(binary_image)
            
            # Step 3: Text Recognition for each line
            recognized_lines = []
            for i, line_image in enumerate(lines):
                print(f"Processing line {i+1}/{len(lines)}")
                text = self.recognize_text(line_image)
                corrected_text = self.correct_text(text)
                recognized_lines.append(corrected_text)
            
            # Step 4: Combine all lines
            full_text = ' '.join(recognized_lines)
            
            # Step 5: Additional processing
            transliterated_text = self.transliterate_to_iso15919(full_text)
            accent_info = self.detect_accent_marks(full_text)
            
            return {
                'image_path': image_path,
                'total_lines': len(lines),
                'sanskrit_text': full_text,
                'transliterated_text': transliterated_text,
                'accent_info': accent_info,
                'processing_status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'processing_status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def process_folder(self, folder_path: str, output_folder: str = None) -> List[dict]:
        """
        Process all images in a folder
        Main function to use when you have a folder full of Sanskrit manuscript images
        """
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Create output folder if specified
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        image_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            print("No image files found in the folder!")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n--- Processing Image {i}/{len(image_files)} ---")
            result = self.process_single_image(image_path)
            results.append(result)
            
            # Save individual result if output folder is specified
            if output_folder:
                filename = os.path.splitext(os.path.basename(image_path))[0]
                result_file = os.path.join(output_folder, f"{filename}_result.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Save combined results
        if output_folder:
            combined_file = os.path.join(output_folder, "all_results.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Create CSV summary
            csv_data = []
            for result in results:
                if result['processing_status'] == 'success':
                    csv_data.append({
                        'image_name': os.path.basename(result['image_path']),
                        'total_lines': result['total_lines'],
                        'sanskrit_text': result['sanskrit_text'][:100] + '...' if len(result['sanskrit_text']) > 100 else result['sanskrit_text'],
                        'transliterated_text': result['transliterated_text'][:100] + '...' if len(result['transliterated_text']) > 100 else result['transliterated_text'],
                        'has_vedic_accents': result['accent_info']['has_vedic_accents'],
                        'processing_status': result['processing_status']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_file = os.path.join(output_folder, "results_summary.csv")
                df.to_csv(csv_file, index=False, encoding='utf-8')
                print(f"Results saved to {output_folder}")
        
        return results

def main():
    """
    Main function - Easy to use interface
    Just change the folder_path to point to your images!
    """
    # Initialize the OCR system
    ocr = SanskritOCR()
    
    # CHANGE THIS PATH to your folder containing Sanskrit manuscript images
    folder_path = r"d:\Projects\Sanskrit-OCR\maitrayani_dataset\line_images"
    output_folder = r"d:\Projects\Sanskrit-OCR\ocr_results"
    
    print(f"Processing images from: {folder_path}")
    print(f"Results will be saved to: {output_folder}")
    
    # Process all images
    results = ocr.process_folder(folder_path, output_folder)
    
    # Print summary
    successful = sum(1 for r in results if r['processing_status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total images processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Show sample results
    if successful > 0:
        print(f"\n=== SAMPLE RESULTS ===")
        for result in results[:3]:  # Show first 3 successful results
            if result['processing_status'] == 'success':
                print(f"Image: {os.path.basename(result['image_path'])}")
                print(f"Sanskrit: {result['sanskrit_text'][:100]}...")
                print(f"Transliterated: {result['transliterated_text'][:100]}...")
                print("-" * 50)

if __name__ == "__main__":
    main()