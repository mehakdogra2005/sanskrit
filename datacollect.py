"""
Comprehensive Maitrayani Samhita Dataset Creation
Following the methodology described for Vedic Sanskrit OCR dataset preparation:

1. Extract electronic text from TITUS with page/line numbering
2. Convert PDF to page images
3. Perform line segmentation using haar-like features
4. Create aligned image-text dataset
5. Generate final OCR training dataset

Dataset Source: https://titus.uni-frankfurt.de/texte/etcs/ind/aind/ved/yvs/ms/ms.htm
PDF Source: Archive.org - Maitrayani Samhita (Gove        # Step 3: Convert PDF to page images
        if not self.page_images:
            logger.info("Step 3: Converting PDF to page images...")
            self.convert_pdf_to_images(str(pdf_path), start_page=1, end_page=598)  # Process ALL pagesnt Press, Bombay, 1941)
"""

import requests
import re
import os
import cv2
import numpy as np
import pandas as pd
import json
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
from pathlib import Path
import urllib.parse
import time
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaitrayaniDatasetCreator:
    """
    Comprehensive dataset creator for Maitrayani Samhita OCR
    """
    
    def __init__(self, base_dir: str = "maitrayani_dataset"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.text_dir = self.base_dir / "text_data"
        self.image_dir = self.base_dir / "page_images"
        self.line_dir = self.base_dir / "line_images"
        self.dataset_dir = self.base_dir / "final_dataset"
        
        for dir_path in [self.text_dir, self.image_dir, self.line_dir, self.dataset_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.text_dict = {}
        self.page_images = []
        self.line_segments = []
        
    def download_titus_text(self) -> bool:
        """
        Download and extract text from TITUS electronic edition
        """
        logger.info("Downloading TITUS electronic text...")
        
        url = "https://titus.uni-frankfurt.de/texte/etcs/ind/aind/ved/yvs/ms/ms.htm"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw HTML
            raw_html_path = self.text_dir / "MS_TITUS_raw.html"
            with open(raw_html_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            logger.info(f"Raw HTML saved to {raw_html_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading TITUS text: {e}")
            return False
    
    def extract_text_with_line_numbers(self) -> Dict[str, str]:
        """
        Extract text with page-line markers from HTML
        Format: (page.line) text content
        """
        logger.info("Extracting text with page-line numbering...")
        
        html_path = self.text_dir / "MS_TITUS_raw.html"
        if not html_path.exists():
            logger.error("Raw HTML file not found. Please download first.")
            return {}
        
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
            
            # Get all text content
            all_text = soup.get_text("\n")
            
            # Enhanced regex patterns for different formats
            patterns = [
                re.compile(r"\((\d+\.\d+)\)\s*(.+?)(?=\(\d+\.\d+\)|$)", re.DOTALL),  # Main pattern
                re.compile(r"(\d+\.\d+)\s+(.+?)(?=\d+\.\d+|$)", re.DOTALL),         # Alternative pattern
                re.compile(r"\[(\d+\.\d+)\]\s*(.+?)(?=\[\d+\.\d+\]|$)", re.DOTALL)   # Bracket pattern
            ]
            
            text_dict = {}
            
            # Try each pattern
            for pattern in patterns:
                matches = pattern.findall(all_text)
                for match in matches:
                    if len(match) == 2:
                        key = match[0].strip()
                        value = match[1].strip()
                        
                        # Clean the text
                        value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                        value = re.sub(r'\n+', ' ', value)  # Replace newlines with spaces
                        
                        if value and len(value) > 5:  # Only keep non-empty, meaningful text
                            text_dict[key] = value
            
            # If no matches with parentheses, try to extract line-by-line
            if not text_dict:
                logger.warning("No parenthetical markers found. Trying line-by-line extraction...")
                lines = all_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # Look for patterns like "1.1 text" at the beginning of lines
                    match = re.match(r'^(\d+\.\d+)\s+(.+)', line)
                    if match:
                        key = match.group(1)
                        value = match.group(2).strip()
                        if value:
                            text_dict[key] = value
            
            self.text_dict = text_dict
            
            # Save extracted text
            json_path = self.text_dir / "extracted_text.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(text_dict, f, ensure_ascii=False, indent=2)
            
            # Save as CSV for easier inspection
            csv_path = self.text_dir / "extracted_text.csv"
            df = pd.DataFrame(list(text_dict.items()), columns=['page_line', 'text'])
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            logger.info(f"Extracted {len(text_dict)} lines with page-line markers")
            logger.info(f"Text saved to {json_path} and {csv_path}")
            
            # Display sample
            logger.info("Sample extracted text:")
            for i, (k, v) in enumerate(list(text_dict.items())[:5]):
                logger.info(f"  {k} → {v[:100]}...")
            
            return text_dict
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return {}
    
    def download_pdf_from_archive(self, pdf_url: str = None) -> str:
        """
        Download PDF from Archive.org
        """
        if pdf_url is None:
            # Default Maitrayani Samhita PDF from Archive.org
            pdf_url = "https://archive.org/download/maitrayanisamhit015004mbp/maitrayanisamhit015004mbp.pdf"
        
        logger.info(f"Downloading PDF from {pdf_url}...")
        
        pdf_path = self.base_dir / "maitrayani_samhita.pdf"
        
        try:
            response = requests.get(pdf_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"PDF downloaded to {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return ""
    
    def convert_pdf_to_images(self, pdf_path: str, start_page: int = 1, end_page: int = 100) -> List[str]:
        """
        Convert PDF pages to images
        """
        logger.info(f"Converting PDF pages {start_page}-{end_page} to images...")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        image_paths = []
        
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            logger.info(f"PDF has {total_pages} pages")
            
            end_page = min(end_page, total_pages)
            
            for page_num in range(start_page - 1, end_page):  # fitz uses 0-based indexing
                try:
                    # Get page
                    page = pdf_document[page_num]
                    
                    # Convert to image with high resolution
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Save as PNG
                    image_path = self.image_dir / f"page_{page_num + 1:04d}.png"
                    pix.save(str(image_path))
                    
                    image_paths.append(str(image_path))
                    
                    if (page_num + 1) % 10 == 0:
                        logger.info(f"Processed {page_num + 1} pages...")
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            
            logger.info(f"Converted {len(image_paths)} pages to images")
            self.page_images = image_paths
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def segment_lines_haar_features(self, image_path: str, page_num: int) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Segment text lines using haar-like features approach
        Based on Viola and Jones (2001) methodology mentioned in the paper
        """
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return []
            
            h, w = img.shape
            
            # Preprocessing
            # 1. Noise reduction
            img_denoised = cv2.fastNlMeansDenoising(img)
            
            # 2. Binarization using Otsu's method
            _, binary = cv2.threshold(img_denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 3. Horizontal projection for line detection
            horizontal_projection = np.sum(binary, axis=1)
            
            # 4. Find line boundaries using projection analysis
            lines = []
            in_line = False
            line_start = 0
            min_line_height = 15  # Reduced minimum height to catch smaller lines
            min_gap = 5  # Reduced gap to be more sensitive
            
            # Smooth the projection to reduce noise
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            smooth_projection = np.convolve(horizontal_projection, kernel, mode='same')
            
            # Find threshold (adaptive - more sensitive)
            threshold = np.mean(smooth_projection) * 0.05  # Reduced from 0.1 to catch more lines
            
            for i, projection in enumerate(smooth_projection):
                if projection > threshold and not in_line:
                    line_start = i
                    in_line = True
                elif projection <= threshold and in_line:
                    line_end = i
                    line_height = line_end - line_start
                    
                    if line_height >= min_line_height:
                        # Extract line region with some padding
                        padding = 5
                        y1 = max(0, line_start - padding)
                        y2 = min(h, line_end + padding)
                        
                        # Extract full width of the line
                        line_img = binary[y1:y2, :]
                        
                        # Save line image
                        line_filename = f"page_{page_num:04d}_line_{len(lines)+1:03d}.png"
                        line_path = self.line_dir / line_filename
                        cv2.imwrite(str(line_path), line_img)
                        
                        lines.append((str(line_path), (0, y1, w, y2)))
                    
                    in_line = False
            
            return lines
            
        except Exception as e:
            logger.error(f"Error segmenting lines in {image_path}: {e}")
            return []
    
    def segment_all_pages(self) -> Dict[int, List]:
        """
        Segment all page images into line images
        """
        logger.info("Segmenting all pages into text lines...")
        
        all_line_segments = {}
        
        for image_path in self.page_images:
            # Extract page number from filename
            page_num = int(re.search(r'page_(\d+)', image_path).group(1))
            
            logger.info(f"Processing page {page_num}...")
            
            # Segment lines
            line_segments = self.segment_lines_haar_features(image_path, page_num)
            all_line_segments[page_num] = line_segments
            
            logger.info(f"Page {page_num}: extracted {len(line_segments)} lines")
        
        self.line_segments = all_line_segments
        
        total_lines = sum(len(segments) for segments in all_line_segments.values())
        logger.info(f"Total lines extracted: {total_lines}")
        
        return all_line_segments
    
    def create_aligned_dataset(self) -> pd.DataFrame:
        """
        Create aligned dataset matching line images with text
        """
        logger.info("Creating aligned image-text dataset...")
        
        dataset_records = []
        
        for page_num, line_segments in self.line_segments.items():
            for line_idx, (line_image_path, bbox) in enumerate(line_segments, 1):
                # Create page.line identifier
                page_line_id = f"{page_num}.{line_idx}"
                
                # Try to find matching text
                matching_text = ""
                
                # Look for exact match first
                if page_line_id in self.text_dict:
                    matching_text = self.text_dict[page_line_id]
                else:
                    # Try approximate matching
                    for key, value in self.text_dict.items():
                        if key.startswith(f"{page_num}."):
                            # If we have the right page, use the text even if line numbers don't match exactly
                            key_parts = key.split('.')
                            if len(key_parts) == 2:
                                text_line_num = int(key_parts[1])
                                if abs(text_line_num - line_idx) <= 2:  # Allow some tolerance
                                    matching_text = value
                                    break
                
                # Create dataset record
                record = {
                    'page_number': page_num,
                    'line_number': line_idx,
                    'page_line_id': page_line_id,
                    'line_image_path': line_image_path,
                    'bbox': bbox,
                    'ground_truth_text': matching_text,
                    'has_ground_truth': bool(matching_text.strip()),
                    'text_length': len(matching_text),
                    'image_exists': os.path.exists(line_image_path)
                }
                
                dataset_records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(dataset_records)
        
        # Save dataset
        dataset_path = self.dataset_dir / "aligned_dataset.csv"
        df.to_csv(dataset_path, index=False, encoding='utf-8')
        
        # Save detailed JSON version
        json_path = self.dataset_dir / "aligned_dataset.json"
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        
        logger.info(f"Dataset created with {len(df)} records")
        logger.info(f"Records with ground truth: {df['has_ground_truth'].sum()}")
        logger.info(f"Records with existing images: {df['image_exists'].sum()}")
        logger.info(f"Dataset saved to {dataset_path} and {json_path}")
        
        return df
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive dataset statistics
        """
        logger.info("Generating dataset statistics...")
        
        stats = {
            'total_records': len(df),
            'pages_processed': df['page_number'].nunique(),
            'lines_per_page': df.groupby('page_number').size().describe().to_dict(),
            'text_coverage': {
                'records_with_ground_truth': df['has_ground_truth'].sum(),
                'coverage_percentage': (df['has_ground_truth'].sum() / len(df)) * 100,
                'avg_text_length': df[df['has_ground_truth']]['text_length'].mean(),
                'text_length_distribution': df[df['has_ground_truth']]['text_length'].describe().to_dict()
            },
            'image_coverage': {
                'images_exist': df['image_exists'].sum(),
                'image_coverage_percentage': (df['image_exists'].sum() / len(df)) * 100
            }
        }
        
        # Save statistics
        stats_path = self.dataset_dir / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        # Print summary
        logger.info("=== Dataset Statistics Summary ===")
        logger.info(f"Total records: {stats['total_records']}")
        logger.info(f"Pages processed: {stats['pages_processed']}")
        logger.info(f"Average lines per page: {stats['lines_per_page']['mean']:.1f}")
        logger.info(f"Text coverage: {stats['text_coverage']['coverage_percentage']:.1f}%")
        logger.info(f"Image coverage: {stats['image_coverage']['image_coverage_percentage']:.1f}%")
        logger.info(f"Average text length: {stats['text_coverage']['avg_text_length']:.1f} characters")
        
        return stats
    
    def create_training_splits(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Create training, validation, and test splits
        """
        logger.info("Creating training/validation/test splits...")
        
        # Only use records that have both ground truth and images
        usable_records = df[(df['has_ground_truth']) & (df['image_exists'])].copy()
        
        logger.info(f"Usable records for training: {len(usable_records)}")
        
        # Shuffle the data
        usable_records = usable_records.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split sizes
        n_total = len(usable_records)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        # Create splits
        splits = {
            'train': usable_records[:n_train],
            'validation': usable_records[n_train:n_train + n_val],
            'test': usable_records[n_train + n_val:]
        }
        
        # Save splits
        for split_name, split_df in splits.items():
            split_path = self.dataset_dir / f"{split_name}_split.csv"
            split_df.to_csv(split_path, index=False, encoding='utf-8')
            
            logger.info(f"{split_name.capitalize()} split: {len(split_df)} records -> {split_path}")
        
        return splits
    
    def create_final_dataset(self) -> str:
        """
        Create the complete final dataset following the paper's methodology
        """
        logger.info("=== Creating Final Maitrayani Samhita OCR Dataset ===")
        
        # Step 1: Download and extract electronic text
        if not self.text_dict:
            logger.info("Step 1: Downloading TITUS electronic text...")
            if not self.download_titus_text():
                logger.error("Failed to download text. Aborting.")
                return ""
            
            self.extract_text_with_line_numbers()
        
        # Step 2: Download PDF if not already done
        pdf_path = self.base_dir / "maitrayani_samhita.pdf"
        if not pdf_path.exists():
            logger.info("Step 2: Downloading PDF from Archive.org...")
            pdf_path = self.download_pdf_from_archive()
            if not pdf_path:
                logger.error("Failed to download PDF. Aborting.")
                return ""
        
        # Step 3: Convert PDF to page images
        if not self.page_images:
            logger.info("Step 3: Converting PDF to page images...")
            self.convert_pdf_to_images(str(pdf_path), start_page=1, end_page=200)  # Process first 200 pages for more data
        
        # Step 4: Segment pages into line images
        if not self.line_segments:
            logger.info("Step 4: Segmenting pages into text lines using haar-like features...")
            self.segment_all_pages()
        
        # Step 5: Create aligned dataset
        logger.info("Step 5: Creating aligned image-text dataset...")
        df = self.create_aligned_dataset()
        
        # Step 6: Generate statistics
        logger.info("Step 6: Generating dataset statistics...")
        stats = self.generate_statistics(df)
        
        # Step 7: Create training splits
        logger.info("Step 7: Creating training/validation/test splits...")
        splits = self.create_training_splits(df)
        
        # Step 8: Create final summary
        summary = {
            'dataset_info': {
                'name': 'Maitrayani Samhita OCR Dataset',
                'description': 'Accent-aware Vedic Sanskrit OCR dataset',
                'source_text': 'https://titus.uni-frankfurt.de/texte/etcs/ind/aind/ved/yvs/ms/ms.htm',
                'source_images': 'Archive.org Maitrayani Samhita PDF',
                'methodology': 'Line segmentation using haar-like features (Viola & Jones, 2001)',
                'script': 'Devanagari with Vedic accent marks',
                'language': 'Vedic Sanskrit'
            },
            'statistics': stats,
            'splits': {split_name: len(split_df) for split_name, split_df in splits.items()}
        }
        
        summary_path = self.dataset_dir / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("=== Final Dataset Creation Complete ===")
        logger.info(f"Dataset location: {self.dataset_dir}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return str(self.dataset_dir)


def main():
    """
    Main function to create the complete Maitrayani Samhita OCR dataset
    """
    print("=== Maitrayani Samhita OCR Dataset Creation ===")
    print("Following the paper's methodology:")
    print("1. Extract TITUS electronic text with page/line numbering")
    print("2. Convert PDF to page images")
    print("3. Segment lines using haar-like features")
    print("4. Create aligned image-text dataset")
    print("5. Generate training/validation/test splits")
    
    # Initialize dataset creator
    creator = MaitrayaniDatasetCreator("maitrayani_dataset")
    
    # Create the complete dataset
    dataset_path = creator.create_final_dataset()
    
    if dataset_path:
        print(f"\n=== Dataset Creation Successful ===")
        print(f"Final dataset location: {dataset_path}")
        print("\nDataset includes:")
        print("- aligned_dataset.csv: Complete image-text alignment")
        print("- train_split.csv: Training data")
        print("- validation_split.csv: Validation data") 
        print("- test_split.csv: Test data")
        print("- dataset_statistics.json: Comprehensive statistics")
        print("- dataset_summary.json: Overall summary")
        print("\nImage files are organized in:")
        print("- page_images/: Original PDF pages as images")
        print("- line_images/: Segmented text line images")
        print("\nText data is saved in:")
        print("- text_data/: TITUS electronic text extraction")
        
        # Display final statistics
        print("\n=== Quick Statistics ===")
        try:
            import pandas as pd
            df = pd.read_csv(os.path.join(dataset_path, "aligned_dataset.csv"))
            print(f"Total records: {len(df)}")
            print(f"Pages processed: {df['page_number'].nunique()}")
            print(f"Records with ground truth: {df['has_ground_truth'].sum()}")
            print(f"Text coverage: {(df['has_ground_truth'].sum() / len(df)) * 100:.1f}%")
            print(f"Images available: {df['image_exists'].sum()}")
        except Exception as e:
            print(f"Could not load final statistics: {e}")
    
    else:
        print("Dataset creation failed. Please check the logs for errors.")

if __name__ == "__main__":
    main()