from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import imageio_ffmpeg

from .utils import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    build_source_label,
    ensure_project_directories,
    sanitize_filename,
    session_directory,
)


@dataclass
class PreparedSource:
    source_reference: str
    media_path: str
    audio_path: str
    source_label: str


class AudioProcessingError(RuntimeError):
    """Raised when a source cannot be converted into a usable audio file."""


def prepare_audio_source(
    video_file: str | None,
    session_id: str,
) -> PreparedSource:
    ensure_project_directories()

    cleaned_file = (video_file or "").strip()

    if not cleaned_file:
        raise AudioProcessingError("Upload a video file to begin.")

    input_path = Path(cleaned_file)
    if not input_path.exists():
        raise AudioProcessingError(f"Uploaded file does not exist: {input_path}")

    media_path = input_path
    source_reference = str(input_path.resolve())

    source_label = build_source_label(source_reference)
    output_dir = session_directory(session_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{sanitize_filename(source_label)}_mono16k.wav"

    extract_audio_to_wav(media_path, audio_path)

    return PreparedSource(
        source_reference=source_reference,
        media_path=str(media_path),
        audio_path=str(audio_path),
        source_label=source_label,
    )


def extract_audio_to_wav(
    input_media_path: str | Path,
    output_audio_path: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> str:
    input_path = Path(input_media_path)
    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        (
            ffmpeg.input(str(input_path))
            .output(
                str(output_path),
                ac=TARGET_CHANNELS,
                ar=sample_rate,
                format="wav",
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(cmd=imageio_ffmpeg.get_ffmpeg_exe(), capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        message = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise AudioProcessingError(f"FFmpeg failed to extract audio: {message}") from exc

    return str(output_path)
