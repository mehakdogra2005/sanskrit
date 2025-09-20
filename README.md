# Sanskrit OCR Setup Guide

## Quick Start (Windows)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install Tesseract OCR (Backup OCR Engine)
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it (remember the installation path)
3. Add Tesseract to your system PATH or set it in the code

### Step 3: Prepare Your Images
- Put all your Sanskrit manuscript images in one folder
- Supported formats: PNG, JPG, JPEG, TIFF, BMP

### Step 4: Run the OCR System

#### Option A: Use the Easy Demo
```bash
python easy_demo.py
```

#### Option B: Use Main Script
```bash
python model2.py
```

#### Option C: Custom Processing
```python
from model2 import SanskritOCR

# Initialize OCR system
ocr = SanskritOCR()

# Process single image
result = ocr.process_single_image("path/to/your/image.png")

# Process entire folder
results = ocr.process_folder("path/to/your/images/folder", "path/to/output/folder")
```

## System Architecture

```
Sanskrit Manuscript Image
         ↓
1. PREPROCESSING
   - Convert to grayscale
   - Remove noise (old paper artifacts)
   - Binarize (pure black text, white background)
         ↓
2. LINE SEGMENTATION
   - Separate each text line
   - Mathematical approach using horizontal projection
         ↓
3. TEXT RECOGNITION
   - Microsoft TrOCR Model (Primary)
   - Tesseract OCR (Fallback)
         ↓
4. POST-PROCESSING
   - Text correction (fix common OCR errors)
   - Transliteration (Sanskrit → English letters)
   - Accent detection (Vedic pronunciation marks)
         ↓
5. OUTPUT
   - Sanskrit text
   - Transliterated text
   - Analysis results
```

## Output Files

When you process images, the system creates:

1. **all_results.json** - Complete results for all images
2. **results_summary.csv** - Summary table (easy to open in Excel)
3. **{image_name}_result.json** - Individual result for each image

## Features Explained

### 1. Image Preprocessing
- **Grayscale Conversion**: Makes processing faster and more accurate
- **Noise Reduction**: Removes dots, scratches from old manuscripts
- **Binarization**: Creates pure black text on white background

### 2. Line Segmentation
- **Horizontal Projection**: Counts black pixels in each row
- **Automatic Detection**: Finds where each text line starts and ends
- **Filter Small Artifacts**: Ignores tiny marks that aren't text

### 3. Text Recognition
- **Primary**: Microsoft TrOCR (transformer-based AI model)
- **Fallback**: Tesseract with Sanskrit configuration
- **Smart Processing**: Converts image pixels to actual text

### 4. Text Correction
- Fixes common OCR mistakes (rn→m, ii→ī, 0→ो, etc.)
- Sanskrit-specific error patterns

### 5. Transliteration
- Converts Devanagari script to Roman letters
- Uses ISO 15919 international standard
- Preserves pronunciation information

### 6. Accent Detection
- Finds Vedic accent marks (udatta, anudatta)
- Important for Sanskrit pronunciation
- Counts and analyzes accent patterns

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **ImportError: No module named 'transformers'**
   ```bash
   pip install transformers torch
   ```

3. **Tesseract not found**
   - Install Tesseract OCR from official website
   - Add to system PATH or update code with installation path

4. **Out of memory errors**
   - Process smaller batches of images
   - Use smaller images or resize them

5. **Poor OCR results**
   - Ensure images have good contrast
   - Try different preprocessing parameters
   - Use higher resolution scans

### Performance Tips

1. **For better accuracy:**
   - Use high-resolution images (300 DPI or higher)
   - Ensure good contrast between text and background
   - Clean scanned images manually if needed

2. **For faster processing:**
   - Resize very large images
   - Process in smaller batches
   - Use GPU if available (automatic with PyTorch)

## Customization

You can modify the system by:

1. **Changing OCR models**: Update the TrOCR model name
2. **Adjusting preprocessing**: Modify image processing parameters
3. **Adding corrections**: Extend the text correction dictionary
4. **Custom transliteration**: Modify the character mapping

## Dataset Integration

This OCR system works perfectly with your existing dataset structure:
- Input: Images from `maitrayani_dataset/line_images/`
- Output: Structured text data for training/analysis
- Format: JSON and CSV for easy analysis

Happy OCR processing! 🚀
