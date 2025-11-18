import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

dataset_root = Path(r"D:\Projects\Sanskrit-OCR\Sanskrit-helper\OCR-Images-Annotation")
csv_path = dataset_root / "dataset.csv"

data = []

for subdir, dirs, files in os.walk(dataset_root):
    for file in tqdm(files, desc=f"Scanning {subdir}"):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = Path(subdir) / file
            txt_path = img_path.with_suffix('.txt')

            if txt_path.exists():
                try:
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                    data.append({
                        "image_path": str(img_path.resolve()),
                        "text": text
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {txt_path}: {e}")
            else:
                print(f"⚠️ Missing text file for {img_path.name}")

# Save to CSV
pd.DataFrame(data).to_csv(csv_path, index=False, encoding='utf-8')

print(f"\n✅ CSV file created successfully!")
print(f"📄 Saved at: {csv_path}")
print(f"📊 Total samples found: {len(data)}")
