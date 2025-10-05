import os
import csv
from pathlib import Path
import pandas as pd
import kagglehub

# ==== CONFIGURATION ====
# Update this path if your dataset folder name is different
DATA_DIR = Path("jaiahuja/skin-cancer-detection")

# Accept common image formats
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def scan_folder(root_dir: Path):
    """Return list of (relative_path, label) pairs for images in subfolders."""
    data = []
    for label_dir in sorted(root_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_file in sorted(label_dir.rglob('*')):
            if img_file.suffix.lower() in IMAGE_EXTS:
                # Use relative path for Kaggle compatibility
                rel_path = str(img_file.relative_to(DATA_DIR.parent))
                data.append((rel_path, label))
    return data


def write_csv(data, out_csv):
    """Write (filepath, label) pairs to a CSV."""
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        for row in data:
            writer.writerow(row)
    print(f"✅ Wrote {len(data)} entries → {out_csv}")


# ==== MAIN PROCESS ====
for split_name in ['Train', 'Test', 'train', 'test']:
    split_dir = DATA_DIR / split_name
    if split_dir.exists():
        print(f"📂 Scanning {split_dir} ...")
        data = scan_folder(split_dir)
        csv_name = f"{split_name.lower()}_labels.csv"
        write_csv(data, csv_name)

        # Quick preview
        df = pd.read_csv(csv_name)
        print(f"🧾 {csv_name} — {len(df)} rows")
        print(df.head(), "\n")
