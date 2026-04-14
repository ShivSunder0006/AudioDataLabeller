from __future__ import annotations

import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .utils import (
    DATASET_DIR,
    DATASET_METADATA_PATH,
    EMOTIONS,
    TARGET_SAMPLE_RATE,
    TEMP_CLIPS_DIR,
    build_source_label,
    ensure_project_directories,
    format_seconds,
    sanitize_filename,
    session_directory,
)


class SegmentLike(Protocol):
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass
class ClipRecord:
    clip_id: str
    session_id: str
    source_video: str
    clip_path: str
    filename: str
    timestamp: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    status: str = "pending"
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def remove_bgm(samples: np.ndarray, sr: int) -> np.ndarray:
    """Gently reduce background music while preserving speech clarity.

    Uses a light Harmonic-Percussive separation blended with the original
    signal, plus a high-pass filter to cut sub-bass rumble from BGM.
    """
    # High-pass filter at 85 Hz to remove sub-bass/rumble from BGM
    filtered = librosa.effects.preemphasis(samples, coef=0.97)

    # Gentle HPSS: low margin keeps voice intact
    harmonic, _percussive = librosa.effects.hpss(filtered, margin=1.0)

    # Blend: keep mostly harmonic but mix back some original for naturalness
    # 65% cleaned harmonic + 35% original preserves voice texture
    blended = 0.65 * harmonic + 0.35 * filtered

    # Normalize to prevent clipping
    peak = np.max(np.abs(blended))
    if peak > 0:
        blended = blended / peak * 0.95

    return blended


class ClipManager:
    def __init__(self, dataset_dir: Path = DATASET_DIR, temp_dir: Path = TEMP_CLIPS_DIR) -> None:
        ensure_project_directories()
        self.dataset_dir = dataset_dir
        self.temp_dir = temp_dir
        self.manifest_path = self.temp_dir / "manifest.csv"
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        if self.manifest_path.exists():
            return

        pd.DataFrame(
            columns=[
                "clip_id",
                "session_id",
                "source_video",
                "clip_path",
                "filename",
                "timestamp",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "status",
                "label",
                "created_at",
                "updated_at",
            ]
        ).to_csv(self.manifest_path, index=False)

    def create_clips_from_segments(
        self,
        source_audio_path: str | Path,
        source_video: str,
        session_id: str,
        segments: Iterable[SegmentLike],
    ) -> list[ClipRecord]:
        # Load full source audio at target sample rate for consistent processing
        full_samples, sr = librosa.load(
            str(source_audio_path), sr=TARGET_SAMPLE_RATE, mono=True
        )

        output_dir = session_directory(session_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_records: list[ClipRecord] = []
        source_stub = sanitize_filename(build_source_label(source_video), fallback="source")

        for index, segment in enumerate(segments, start=1):
            # Slice the segment from the full audio
            start_sample = int(segment.start_seconds * sr)
            end_sample = int(segment.end_seconds * sr)
            clip_samples = full_samples[start_sample:end_sample]

            if clip_samples.size == 0:
                continue

            # Remove background music from the clip
            clean_samples = remove_bgm(clip_samples, sr)

            clip_id = f"{session_id}_{index:03d}_{uuid.uuid4().hex[:6]}"
            filename = f"{source_stub}_{clip_id}.wav"
            clip_path = output_dir / filename

            # Export as 16kHz mono WAV
            sf.write(str(clip_path), clean_samples, sr, subtype="PCM_16")

            created_records.append(
                ClipRecord(
                    clip_id=clip_id,
                    session_id=session_id,
                    source_video=source_video,
                    clip_path=str(clip_path),
                    filename=filename,
                    timestamp=f"{format_seconds(segment.start_seconds)} - {format_seconds(segment.end_seconds)}",
                    start_seconds=round(segment.start_seconds, 3),
                    end_seconds=round(segment.end_seconds, 3),
                    duration_seconds=round(segment.duration_seconds, 3),
                )
            )

        self._append_to_manifest(created_records)
        return created_records

    def mark_skipped(self, clip: dict[str, Any]) -> dict[str, Any]:
        clip["status"] = "skipped"
        self._update_manifest_row(clip)
        return clip

    def label_clip(self, clip: dict[str, Any], label: str) -> dict[str, Any]:
        if label not in EMOTIONS:
            raise ValueError(f"Unsupported label: {label}")

        source_path = Path(clip["clip_path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Clip file does not exist: {source_path}")

        target_dir = self.dataset_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        if clip.get("label") == label and source_path.parent.resolve() == target_dir.resolve():
            clip["status"] = "labeled"
            self._update_manifest_row(clip)
            self._upsert_dataset_metadata(clip)
            return clip

        target_path = self._unique_destination(target_dir / source_path.name)

        shutil.move(str(source_path), target_path)

        clip["clip_path"] = str(target_path)
        clip["filename"] = target_path.name
        clip["label"] = label
        clip["status"] = "labeled"

        self._update_manifest_row(clip)
        self._upsert_dataset_metadata(clip)
        return clip

    def undo_label(
        self,
        clip: dict[str, Any],
        previous_path: str,
        previous_status: str,
        previous_label: str | None,
        previous_filename: str,
    ) -> dict[str, Any]:
        current_path = Path(clip["clip_path"])
        destination = Path(previous_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if current_path.exists():
            restored_path = self._unique_destination(destination)
            shutil.move(str(current_path), restored_path)
        else:
            restored_path = destination

        clip["clip_path"] = str(restored_path)
        clip["filename"] = restored_path.name
        clip["status"] = previous_status
        clip["label"] = previous_label

        self._update_manifest_row(clip)
        self._sync_dataset_metadata_after_undo(clip)
        return clip

    def _append_to_manifest(self, records: list[ClipRecord]) -> None:
        if not records:
            return

        now = datetime.utcnow().isoformat()
        rows = []
        for record in records:
            payload = record.to_dict()
            payload["created_at"] = now
            payload["updated_at"] = now
            rows.append(payload)

        manifest = pd.read_csv(self.manifest_path)
        manifest = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
        manifest.to_csv(self.manifest_path, index=False)

    def _update_manifest_row(self, clip: dict[str, Any]) -> None:
        manifest = pd.read_csv(self.manifest_path)
        if manifest.empty or clip["clip_id"] not in set(manifest["clip_id"].astype(str)):
            self._append_to_manifest([ClipRecord(**clip)])
            return

        mask = manifest["clip_id"].astype(str) == str(clip["clip_id"])
        for column, value in clip.items():
            if column in manifest.columns:
                manifest.loc[mask, column] = value
        manifest.loc[mask, "updated_at"] = datetime.utcnow().isoformat()
        manifest.to_csv(self.manifest_path, index=False)

    def _upsert_dataset_metadata(self, clip: dict[str, Any]) -> None:
        metadata = pd.read_csv(DATASET_METADATA_PATH)
        row = {
            "filename": clip["filename"],
            "label": clip["label"],
            "timestamp": clip["timestamp"],
            "source_video": clip["source_video"],
            "clip_id": clip["clip_id"],
            "duration_seconds": clip["duration_seconds"],
            "start_seconds": clip["start_seconds"],
            "end_seconds": clip["end_seconds"],
            "labeled_at": datetime.utcnow().isoformat(),
        }

        if metadata.empty or clip["clip_id"] not in set(metadata["clip_id"].astype(str)):
            metadata = pd.concat([metadata, pd.DataFrame([row])], ignore_index=True)
        else:
            mask = metadata["clip_id"].astype(str) == str(clip["clip_id"])
            for column, value in row.items():
                metadata.loc[mask, column] = value

        metadata.to_csv(DATASET_METADATA_PATH, index=False)

    def _sync_dataset_metadata_after_undo(self, clip: dict[str, Any]) -> None:
        metadata = pd.read_csv(DATASET_METADATA_PATH)
        if metadata.empty or clip["clip_id"] not in set(metadata["clip_id"].astype(str)):
            return

        mask = metadata["clip_id"].astype(str) == str(clip["clip_id"])
        if clip["label"]:
            metadata.loc[mask, "filename"] = clip["filename"]
            metadata.loc[mask, "label"] = clip["label"]
            metadata.loc[mask, "labeled_at"] = datetime.utcnow().isoformat()
        else:
            metadata = metadata.loc[~mask]

        metadata.to_csv(DATASET_METADATA_PATH, index=False)

    @staticmethod
    def _unique_destination(target_path: Path) -> Path:
        if not target_path.exists():
            return target_path

        suffix = target_path.suffix
        stem = target_path.stem
        parent = target_path.parent
        candidate = parent / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
        return candidate
