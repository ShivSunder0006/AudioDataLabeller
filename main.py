# Hugging Face Entry Point
from pathlib import Path
from src.ui import build_demo
from src.utils import DATASET_DIR, TEMP_CLIPS_DIR, ensure_project_directories

print(">>> [STITCH] Initializing via main.py...")
ensure_project_directories()

demo = build_demo()

# Standard HF launch
demo.launch()
