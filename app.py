# Build trigger: V4 - Zero Config
from pathlib import Path
from src.ui import build_demo
from src.utils import DATASET_DIR, TEMP_CLIPS_DIR, ensure_project_directories

# Ensure local directories
ensure_project_directories()

# Build the demo
demo = build_demo()

# Launch with ZERO arguments. 
# Hugging Face Spaces automatically configures networking for this call.
demo.launch()
