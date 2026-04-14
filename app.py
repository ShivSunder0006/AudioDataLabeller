# Build trigger: V3 - Diagnostic Logging
from pathlib import Path
from src.ui import build_demo
from src.utils import DATASET_DIR, TEMP_CLIPS_DIR, ensure_project_directories

print(">>> [STITCH] Starting Audio Dataset Labeler...")
try:
    ensure_project_directories()
    print(f">>> [STITCH] Directories ensured: {DATASET_DIR}, {TEMP_CLIPS_DIR}")
except Exception as e:
    print(f">>> [STITCH] CRITICAL ERROR during directory setup: {e}")

print(">>> [STITCH] Building Gradio interface...")
try:
    demo = build_demo()
    print(">>> [STITCH] Interface built successfully.")
except Exception as e:
    print(f">>> [STITCH] CRITICAL ERROR during interface build: {e}")

print(">>> [STITCH] Launching Gradio...")
# Force HF compatibility
demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True
)
