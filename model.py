"""
🕉️ SANSKRIT OCR MODEL - COMPLETE PIPELINE 🕉️
===============================================

This is a comprehensive Sanskrit OCR (Optical Character Recognition) system
that can read ancient Sanskrit manuscripts and convert them to digital text.

WHAT THIS SYSTEM DOES:
1. 🖼️  Takes images of Sanskrit manuscript pages
2. 🧹 Cleans and preprocesses the images
3. ✂️  Cuts each page into individual text lines
4. 🔤 Recognizes Sanskrit characters in each line
5. ✅ Corrects any mistakes in recognition
6. 🌍 Converts Sanskrit script to English letters (transliteration)
7. 🎵 Detects special accent marks for proper pronunciation
8. 📊 Evaluates how accurate the recognition is

AUTHOR: Sanskrit OCR Project
DATE: 2025
PURPOSE: Digitizing ancient Sanskrit texts for preservation and study
"""

import cv2              # For image processing
import numpy as np      # For numerical operations
import pandas as pd     # For data handling
import os               # For file operations
from pathlib import Path
import json
import re
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch            # For AI/machine learning
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging
from typing import List, Dict, Tuple, Optional
import unicodedata

# Set up logging to track what the program is doing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SanskritOCR:
    """
    🕉️ MAIN SANSKRIT OCR CLASS
    
    This is the main class that handles the entire OCR process.
    Think of it as the "brain" that coordinates all the different steps.
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        """
        Initialize the Sanskrit OCR system
        
        Args:
            model_name: The AI model to use for text recognition
                       (Microsoft's TrOCR is very good for printed text)
        """
        logger.info("🚀 Initializing Sanskrit OCR System...")
        
        # Load the AI model for text recognition
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            logger.info("✅ AI model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load AI model: {e}")
            raise
        
        # Initialize components
        self.image_preprocessor = ImagePreprocessor()
        self.line_segmenter = LineSegmenter()
        self.text_recognizer = TextRecognizer(self.processor, self.model)
        self.text_corrector = TextCorrector()
        self.transliterator = SanskritTransliterator()
        self.accent_detector = AccentDetector()
        self.performance_evaluator = PerformanceEvaluator()
        
        logger.info("🎯 All components initialized!")
    
    def process_single_image(self, image_path: str, ground_truth: str = None) -> Dict:
        """
        Process a single image through the complete OCR pipeline
        
        Args:
            image_path: Path to the image file
            ground_truth: Correct text (if available) for accuracy testing
            
        Returns:
            Dictionary with all results from each step
        """
        logger.info(f"📸 Processing image: {os.path.basename(image_path)}")
        
        results = {
            'image_path': image_path,
            'ground_truth': ground_truth,
            'steps': {}
        }
        
        try:
            # STEP 1: Load and preprocess the image
            logger.info("🧹 Step 1: Cleaning and preprocessing image...")
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            processed_image = self.image_preprocessor.preprocess_image_for_ocr(original_image)
            results['steps']['preprocessing'] = {
                'status': 'success',
                'original_shape': original_image.shape,
                'processed_shape': processed_image.shape
            }
            
            # STEP 2: Segment lines
            logger.info("✂️ Step 2: Segmenting text lines...")
            line_images = self.line_segmenter.segment_lines(processed_image)
            results['steps']['line_segmentation'] = {
                'status': 'success',
                'lines_found': len(line_images)
            }
            
            # STEP 3: Recognize text in each line
            logger.info("🔤 Step 3: Recognizing text...")
            recognized_lines = []
            for i, line_img in enumerate(line_images):
                try:
                    text = self.text_recognizer.recognize_text(line_img)
                    recognized_lines.append(text)
                    logger.info(f"   Line {i+1}: {text[:50]}...")
                except Exception as e:
                    logger.warning(f"   Failed to recognize line {i+1}: {e}")
                    recognized_lines.append("")
            
            raw_text = '\n'.join(recognized_lines)
            results['steps']['text_recognition'] = {
                'status': 'success',
                'raw_text': raw_text,
                'lines_recognized': len([l for l in recognized_lines if l.strip()])
            }
            
            # STEP 4: Correct text errors
            logger.info("✅ Step 4: Correcting recognition errors...")
            corrected_text = self.text_corrector.weighted_edit_distance(raw_text, ground_truth or "")
            results['steps']['text_correction'] = {
                'status': 'success',
                'corrected_text': corrected_text
            }
            
            # STEP 5: Transliterate to English
            logger.info("🌍 Step 5: Transliterating to English...")
            transliterated_text = self.transliterator.transliterate_to_iso15919(corrected_text)
            results['steps']['transliteration'] = {
                'status': 'success',
                'transliterated_text': transliterated_text
            }
            
            # STEP 6: Detect accent marks
            logger.info("🎵 Step 6: Detecting accent marks...")
            accent_info = self.accent_detector.detect_accent_marks(corrected_text)
            results['steps']['accent_detection'] = {
                'status': 'success',
                'accent_marks': accent_info
            }
            
            # STEP 7: Evaluate performance (if ground truth is available)
            if ground_truth:
                logger.info("📊 Step 7: Evaluating accuracy...")
                performance = self.performance_evaluator.evaluate_ocr_performance(
                    corrected_text, ground_truth
                )
                results['steps']['performance_evaluation'] = {
                    'status': 'success',
                    'metrics': performance
                }
            
            results['final_output'] = {
                'sanskrit_text': corrected_text,
                'transliterated_text': transliterated_text,
                'accent_marks': accent_info
            }
            
            logger.info("🎉 Image processing completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            results['error'] = str(e)
        
        return results
    
    def process_dataset(self, dataset_path: str, output_dir: str = "ocr_results") -> Dict:
        """
        Process an entire dataset of images
        
        Args:
            dataset_path: Path to the dataset CSV file
            output_dir: Directory to save results
            
        Returns:
            Summary of processing results
        """
        logger.info(f"📚 Processing entire dataset: {dataset_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"📊 Loaded dataset with {len(df)} records")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return {'error': f"Failed to load dataset: {e}"}
        
        results = []
        successful_processes = 0
        
        # Process each image
        for idx, row in df.iterrows():
            logger.info(f"📸 Processing image {idx + 1}/{len(df)}")
            
            image_path = row.get('image_path', '')
            ground_truth = row.get('ground_truth_text', None)
            
            if not os.path.exists(image_path):
                logger.warning(f"⚠️ Image not found: {image_path}")
                continue
            
            # Process single image
            result = self.process_single_image(image_path, ground_truth)
            
            if 'error' not in result:
                successful_processes += 1
            
            results.append(result)
            
            # Save individual result
            result_file = output_path / f"result_{idx:04d}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Generate summary
        summary = {
            'total_images': len(df),
            'successful_processes': successful_processes,
            'success_rate': (successful_processes / len(df)) * 100,
            'output_directory': str(output_path)
        }
        
        # Save summary
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎯 Dataset processing completed! Success rate: {summary['success_rate']:.1f}%")
        
        return summary

class ImagePreprocessor:
    """
    🧹 IMAGE PREPROCESSING CLASS
    
    This class cleans up images to make them easier for the AI to read.
    It's like cleaning a dirty photo to make the text clearer.
    """
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Clean up the image so the computer can read it better
        
        Steps:
        1. Convert to black and white (grayscale)
        2. Remove noise and scratches from old paper (denoising) 
        3. Make text pure black and background pure white (binarization)
        4. Make the image the right size for the AI model
        
        Args:
            image: The original image as a numpy array
            
        Returns:
            Cleaned image ready for OCR
        """
        logger.info("🧹 Cleaning image for OCR...")
        
        # Step 1: Convert colored image to black and white (grayscale)
        if len(image.shape) == 3:  # If image has colors
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Remove noise/scratches from old paper (denoising)
        # This is like using an eraser to remove spots and marks
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Step 3: Make text pure black and background pure white (binarization)
        # This makes the text stand out clearly
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 4: Make sure text is black on white background
        # Sometimes scans are reversed (white text on black background)
        if np.mean(binary) < 127:  # If image is mostly dark
            binary = cv2.bitwise_not(binary)  # Flip black and white
        
        logger.info("✅ Image preprocessing completed!")
        return binary

class LineSegmenter:
    """
    ✂️ LINE SEGMENTATION CLASS
    
    This class separates each line of text from the full page.
    It's like using a ruler to separate each line of handwritten text.
    """
    
    def segment_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Separate each line of text from the full page image
        
        Method: Horizontal Projection
        - Count black pixels in each row
        - Find boundaries where text lines start and end
        - Extract each line as a separate image
        
        Args:
            image: Preprocessed page image
            
        Returns:
            List of line images
        """
        logger.info("✂️ Segmenting text lines...")
        
        lines = []
        
        # Step 1: Count black pixels in each row (Horizontal Projection)
        horizontal_projection = np.sum(image == 0, axis=1)  # Count black pixels per row
        
        # Step 2: Find where text lines start and end
        # Look for rows with many black pixels (text) vs few black pixels (spaces)
        threshold = np.mean(horizontal_projection) * 0.3  # Adjust threshold as needed
        
        in_line = False
        line_start = 0
        
        for i, pixel_count in enumerate(horizontal_projection):
            # If we find enough black pixels and we're not already in a line
            if pixel_count > threshold and not in_line:
                line_start = i
                in_line = True
            
            # If we don't find enough black pixels and we were in a line
            elif pixel_count <= threshold and in_line:
                line_end = i
                in_line = False
                
                # Extract the line
                if line_end - line_start > 10:  # Only keep lines that are tall enough
                    line_image = image[line_start:line_end, :]
                    
                    # Remove empty columns (left and right margins)
                    vertical_projection = np.sum(line_image == 0, axis=0)
                    left_bound = 0
                    right_bound = len(vertical_projection)
                    
                    # Find left boundary
                    for j, col_pixels in enumerate(vertical_projection):
                        if col_pixels > 0:
                            left_bound = max(0, j - 5)  # Add small margin
                            break
                    
                    # Find right boundary
                    for j in range(len(vertical_projection) - 1, -1, -1):
                        if vertical_projection[j] > 0:
                            right_bound = min(len(vertical_projection), j + 5)  # Add small margin
                            break
                    
                    # Extract the clean line
                    clean_line = line_image[:, left_bound:right_bound]
                    
                    if clean_line.shape[1] > 20:  # Only keep lines that are wide enough
                        lines.append(clean_line)
        
        logger.info(f"✅ Found {len(lines)} text lines")
        return lines

class TextRecognizer:
    """
    🔤 TEXT RECOGNITION CLASS
    
    This class uses AI to convert images of text into actual text.
    It's like having a very smart robot that can read Sanskrit characters.
    """
    
    def __init__(self, processor, model):
        """
        Initialize with the AI model
        """
        self.processor = processor
        self.model = model
    
    def recognize_text(self, line_image: np.ndarray) -> str:
        """
        Convert image of text line into actual text
        
        This uses Microsoft's TrOCR model, which is trained on millions
        of text images and is very good at reading printed text.
        
        Args:
            line_image: Image of a single line of text
            
        Returns:
            The text that was recognized in the image
        """
        try:
            # Convert numpy array to PIL Image (format the AI model expects)
            if len(line_image.shape) == 2:  # If grayscale
                pil_image = Image.fromarray(line_image).convert('RGB')
            else:
                pil_image = Image.fromarray(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
            
            # Process image with AI model
            # Step 1: Convert image to numbers the AI can understand
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
            
            # Step 2: Generate text predictions
            generated_ids = self.model.generate(pixel_values)
            
            # Step 3: Convert AI predictions back to readable text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"❌ Text recognition failed: {e}")
            return ""

class TextCorrector:
    """
    ✅ TEXT CORRECTION CLASS
    
    This class fixes mistakes in the recognized text.
    It's like a spell-checker but specifically for Sanskrit.
    """
    
    def weighted_edit_distance(self, recognized_text: str, reference_text: str) -> str:
        """
        Fix mistakes in recognized text by comparing with correct text
        
        This function looks at what the AI found vs. what should be correct,
        and suggests fixes for common mistakes.
        
        Args:
            recognized_text: What the AI thought it saw
            reference_text: What the text should actually say (if available)
            
        Returns:
            Corrected text
        """
        if not reference_text:
            # If we don't have correct text to compare with,
            # just do basic cleaning
            return self._basic_text_cleanup(recognized_text)
        
        # Compare character by character and fix common mistakes
        corrected = recognized_text
        
        # Common Sanskrit OCR mistakes and their corrections
        corrections = {
            # Common character confusions
            'ऱ': 'र',  # ra variants
            'क़': 'क',  # ka variants  
            'ि': 'ी',  # vowel mark confusions
            'ु': 'ू',  # vowel mark confusions
            'ृ': 'ऋ',  # vowel confusions
            '।।': '॥', # punctuation
            # Add more based on your specific data
        }
        
        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        return corrected
    
    def _basic_text_cleanup(self, text: str) -> str:
        """
        Basic text cleaning without reference
        """
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        text = text.strip()
        return text

class SanskritTransliterator:
    """
    🌍 TRANSLITERATION CLASS
    
    This class converts Sanskrit script to English letters.
    This makes it easier for people who can't read Devanagari
    to understand the pronunciation.
    """
    
    def transliterate_to_iso15919(self, sanskrit_text: str) -> str:
        """
        Convert Sanskrit script to English letters using ISO 15919 standard
        
        This is the international standard for converting Devanagari
        script to Roman letters while preserving pronunciation.
        
        Args:
            sanskrit_text: Text in Devanagari script
            
        Returns:
            Text in Roman letters
        """
        # Mapping of Devanagari characters to Roman letters
        # Based on ISO 15919 international standard
        transliteration_map = {
            # Vowels
            'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī',
            'उ': 'u', 'ऊ': 'ū', 'ऋ': 'ṛ', 'ॠ': 'ṝ',
            'ऌ': 'ḷ', 'ॡ': 'ḹ', 'ए': 'e', 'ऐ': 'ai',
            'ओ': 'o', 'औ': 'au',
            
            # Consonants
            'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'ṅa',
            'च': 'ca', 'छ': 'cha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'ña',
            'ट': 'ṭa', 'ठ': 'ṭha', 'ड': 'ḍa', 'ढ': 'ḍha', 'ण': 'ṇa',
            'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
            'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
            'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va',
            'श': 'śa', 'ष': 'ṣa', 'स': 'sa', 'ह': 'ha',
            
            # Vowel marks (when attached to consonants)
            'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū',
            'ृ': 'ṛ', 'ॄ': 'ṝ', 'ॢ': 'ḷ', 'ॣ': 'ḹ',
            'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
            
            # Special symbols
            'ं': 'ṃ',   # Anusvara
            'ः': 'ḥ',   # Visarga
            '्': '',    # Virama (removes inherent vowel)
            '।': '.',   # Danda (period)
            '॥': '॥',   # Double danda
        }
        
        result = ""
        i = 0
        while i < len(sanskrit_text):
            char = sanskrit_text[i]
            
            # Check for multi-character sequences first
            if i < len(sanskrit_text) - 1:
                two_char = sanskrit_text[i:i+2]
                if two_char in transliteration_map:
                    result += transliteration_map[two_char]
                    i += 2
                    continue
            
            # Single character transliteration
            if char in transliteration_map:
                result += transliteration_map[char]
            else:
                result += char  # Keep character as-is if no mapping found
            
            i += 1
        
        return result

class AccentDetector:
    """
    🎵 ACCENT DETECTION CLASS
    
    This class finds special pronunciation marks in Vedic Sanskrit.
    These marks tell you how to sing or chant the text correctly.
    """
    
    def detect_accent_marks(self, text: str) -> Dict:
        """
        Find and count special pronunciation marks in Vedic Sanskrit
        
        Vedic Sanskrit has special marks that show:
        - Udatta: High tone (raised pitch)
        - Anudatta: Low tone (lowered pitch) 
        - Svarita: Falling tone (starts high, falls low)
        
        Args:
            text: Sanskrit text to analyze
            
        Returns:
            Dictionary with accent mark counts and locations
        """
        accent_info = {
            'udatta': {'count': 0, 'positions': []},      # High tone (not usually marked)
            'anudatta': {'count': 0, 'positions': []},    # Low tone (marked with accent)
            'svarita': {'count': 0, 'positions': []},     # Falling tone (marked with accent)
            'total_accents': 0
        }
        
        # Look for accent marks in the text
        for i, char in enumerate(text):
            # Anudatta mark (low tone) - usually marked with a line below
            if char == '\u0951':  # Vedic tone mark anudatta
                accent_info['anudatta']['count'] += 1
                accent_info['anudatta']['positions'].append(i)
            
            # Svarita mark (falling tone) - usually marked with a line above
            elif char == '\u0952':  # Vedic tone mark svarita
                accent_info['svarita']['count'] += 1
                accent_info['svarita']['positions'].append(i)
        
        accent_info['total_accents'] = (
            accent_info['anudatta']['count'] + 
            accent_info['svarita']['count']
        )
        
        return accent_info

class PerformanceEvaluator:
    """
    📊 PERFORMANCE EVALUATION CLASS
    
    This class tests how well the OCR system is working.
    It compares what the AI found vs. what the text should actually say.
    """
    
    def evaluate_ocr_performance(self, recognized_text: str, ground_truth: str) -> Dict:
        """
        Test how good the OCR system is working
        
        Metrics:
        - CER (Character Error Rate): How many individual letters are wrong
        - WER (Word Error Rate): How many complete words are wrong  
        - CHRF1: Overall quality score (higher = better)
        
        Args:
            recognized_text: What the AI thought it saw
            ground_truth: What the text should actually say
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not ground_truth:
            return {'error': 'No ground truth text provided for evaluation'}
        
        # Calculate Character Error Rate (CER)
        # This shows how many individual letters are wrong
        cer = self._calculate_cer(recognized_text, ground_truth)
        
        # Calculate Word Error Rate (WER)  
        # This shows how many complete words are wrong
        wer = self._calculate_wer(recognized_text, ground_truth)
        
        # Calculate CHRF1 score
        # This is an overall quality score (higher = better)
        chrf1 = self._calculate_chrf1(recognized_text, ground_truth)
        
        performance = {
            'cer': cer,
            'wer': wer, 
            'chrf1': chrf1,
            'quality_assessment': self._assess_quality(cer, wer, chrf1)
        }
        
        return performance
    
    def _calculate_cer(self, recognized: str, reference: str) -> float:
        """Calculate Character Error Rate"""
        if len(reference) == 0:
            return 1.0 if len(recognized) > 0 else 0.0
        
        # Count character differences
        errors = sum(c1 != c2 for c1, c2 in zip(recognized, reference))
        errors += abs(len(recognized) - len(reference))
        
        return errors / len(reference)
    
    def _calculate_wer(self, recognized: str, reference: str) -> float:
        """Calculate Word Error Rate"""
        rec_words = recognized.split()
        ref_words = reference.split()
        
        if len(ref_words) == 0:
            return 1.0 if len(rec_words) > 0 else 0.0
        
        # Count word differences
        errors = sum(w1 != w2 for w1, w2 in zip(rec_words, ref_words))
        errors += abs(len(rec_words) - len(ref_words))
        
        return errors / len(ref_words)
    
    def _calculate_chrf1(self, recognized: str, reference: str) -> float:
        """Calculate CHRF1 score (character-level F1)"""
        if not reference:
            return 0.0
        
        # Simple character-level precision/recall
        rec_chars = set(recognized)
        ref_chars = set(reference)
        
        if not rec_chars and not ref_chars:
            return 1.0
        
        intersection = rec_chars & ref_chars
        precision = len(intersection) / len(rec_chars) if rec_chars else 0
        recall = len(intersection) / len(ref_chars) if ref_chars else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _assess_quality(self, cer: float, wer: float, chrf1: float) -> str:
        """Provide a human-readable quality assessment"""
        if cer < 0.05 and wer < 0.1:
            return "Excellent (Very high accuracy)"
        elif cer < 0.1 and wer < 0.2:
            return "Good (High accuracy)" 
        elif cer < 0.2 and wer < 0.4:
            return "Fair (Moderate accuracy)"
        else:
            return "Poor (Low accuracy - needs improvement)"

# =============================================================================
# 🚀 MAIN EXECUTION FUNCTIONS
# =============================================================================

def process_single_image_demo(image_path: str):
    """
    🎯 DEMO: Process a single image
    
    This is a simple example of how to use the OCR system
    on a single image file.
    """
    print("🕉️ Sanskrit OCR - Single Image Demo")
    print("=" * 50)
    
    # Initialize the OCR system
    ocr = SanskritOCR()
    
    # Process the image
    results = ocr.process_single_image(image_path)
    
    # Display results
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print("📋 RESULTS:")
    print(f"✅ Original image: {results['image_path']}")
    
    if 'final_output' in results:
        output = results['final_output']
        print(f"🔤 Sanskrit text: {output['sanskrit_text'][:100]}...")
        print(f"🌍 English letters: {output['transliterated_text'][:100]}...")
        print(f"🎵 Accent marks found: {output['accent_marks']['total_accents']}")
    
    # Show step-by-step results
    print("\n🔍 STEP-BY-STEP RESULTS:")
    for step_name, step_data in results['steps'].items():
        if step_data['status'] == 'success':
            print(f"✅ {step_name.replace('_', ' ').title()}")
        else:
            print(f"❌ {step_name.replace('_', ' ').title()}")

def process_dataset_demo(dataset_path: str):
    """
    🎯 DEMO: Process an entire dataset
    
    This shows how to process many images at once
    using your dataset CSV file.
    """
    print("🕉️ Sanskrit OCR - Dataset Processing Demo")
    print("=" * 50)
    
    # Initialize the OCR system
    ocr = SanskritOCR()
    
    # Process the entire dataset
    summary = ocr.process_dataset(dataset_path, "ocr_results")
    
    # Display results
    if 'error' in summary:
        print(f"❌ Error: {summary['error']}")
        return
    
    print("📊 PROCESSING SUMMARY:")
    print(f"📁 Total images: {summary['total_images']}")
    print(f"✅ Successfully processed: {summary['successful_processes']}")
    print(f"📈 Success rate: {summary['success_rate']:.1f}%")
    print(f"💾 Results saved to: {summary['output_directory']}")

def main():
    """
    🚀 MAIN FUNCTION
    
    This is the main entry point of the program.
    Choose what you want to do:
    1. Process a single image
    2. Process your entire dataset
    """
    print("🕉️ SANSKRIT OCR SYSTEM")
    print("=" * 30)
    print("Welcome to the Sanskrit OCR System!")
    print("This system can read Sanskrit manuscripts and convert them to digital text.")
    print()
    
    print("Choose an option:")
    print("1. Process a single image")
    print("2. Process entire dataset") 
    print("3. Exit")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        # Single image processing
        image_path = input("Enter path to your image file: ").strip().strip('"')
        if os.path.exists(image_path):
            process_single_image_demo(image_path)
        else:
            print(f"❌ Image not found: {image_path}")
    
    elif choice == "2":
        # Dataset processing
        dataset_path = input("Enter path to your dataset CSV file: ").strip().strip('"')
        if os.path.exists(dataset_path):
            process_dataset_demo(dataset_path)
        else:
            print(f"❌ Dataset file not found: {dataset_path}")
            print("💡 Tip: Make sure you have run create_dataset.py first to create the dataset!")
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()
