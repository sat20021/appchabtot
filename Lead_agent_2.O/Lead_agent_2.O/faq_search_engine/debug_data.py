import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import Config
import glob

print("=" * 60)
print("DEBUG: Data Directory Configuration")
print("=" * 60)
print(f"BASE_DIR: {Config.BASE_DIR}")
print(f"DATA_DIR: {Config.DATA_DIR}")
print(f"DATA_DIR exists: {os.path.exists(Config.DATA_DIR)}")
print(f"DATA_DIR absolute: {os.path.abspath(Config.DATA_DIR)}")
print()

# Test loading a file
if files:
    from faq_app.services.ingestor import DataIngestor
    print("Testing DataIngestor on first file...")
    df = DataIngestor.load_file(files[0])
    if df is not None:
        print(f"SUCCESS: Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
    else:
        print("FAILED: DataIngestor returned None")
