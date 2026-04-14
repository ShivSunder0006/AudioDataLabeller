from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATASET_DIR = PROJECT_ROOT / "dataset"
TEMP_CLIPS_DIR = PROJECT_ROOT / "temp_clips"
SESSIONS_DIR = TEMP_CLIPS_DIR / "sessions"
DATASET_METADATA_PATH = DATASET_DIR / "metadata.csv"

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
FRAME_DURATION_MS = 30
CLIP_MIN_SECONDS = 2.0
CLIP_MAX_SECONDS = 5.0
DEFAULT_VAD_MODE = 2
MERGE_GAP_SECONDS = 0.45
PADDING_MS = 180

EMOTIONS = ("happy", "sad", "angry", "neutral", "fear", "disgust")
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
}


def ensure_project_directories() -> None:
    """Create the required project folders and dataset label buckets."""
    for directory in (DATASET_DIR, TEMP_CLIPS_DIR, SESSIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for emotion in EMOTIONS:
        (DATASET_DIR / emotion).mkdir(parents=True, exist_ok=True)

    if not DATASET_METADATA_PATH.exists():
        DATASET_METADATA_PATH.write_text(
            "filename,label,timestamp,source_video,clip_id,duration_seconds,start_seconds,end_seconds,labeled_at\n",
            encoding="utf-8",
        )


def create_session_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]


def sanitize_filename(value: str, fallback: str = "source") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_source_label(source_reference: str) -> str:
    return sanitize_filename(Path(source_reference.strip()).stem, fallback="uploaded_video")


def session_directory(session_id: str) -> Path:
    return SESSIONS_DIR / session_id
