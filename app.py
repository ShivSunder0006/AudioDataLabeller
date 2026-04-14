from pathlib import Path
from src.ui import build_demo
from src.utils import DATASET_DIR, TEMP_CLIPS_DIR, ensure_project_directories

# Ensure directories exist for the Space
ensure_project_directories()

# Build and launch the UI
# Setting server_name="0.0.0.0" is required for HF Spaces
demo = build_demo()
demo.queue(api_open=False).launch(
    server_name="0.0.0.0",
    show_error=True,
    allowed_paths=[
        str(Path(DATASET_DIR).resolve()),
        str(Path(TEMP_CLIPS_DIR).resolve()),
    ],
)
