from __future__ import annotations

import collections
import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import webrtcvad

from .utils import (
    CLIP_MAX_SECONDS,
    CLIP_MIN_SECONDS,
    DEFAULT_VAD_MODE,
    FRAME_DURATION_MS,
    MERGE_GAP_SECONDS,
    PADDING_MS,
    TARGET_SAMPLE_RATE,
)


@dataclass
class Frame:
    bytes_data: bytes
    timestamp: float
    duration: float


@dataclass
class SpeechSegment:
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    speech_ratio: float
    quality_score: float


def segment_conversational_clips(
    wav_path: str | Path,
    vad_mode: int = DEFAULT_VAD_MODE,
    min_clip_seconds: float = CLIP_MIN_SECONDS,
    max_clip_seconds: float = CLIP_MAX_SECONDS,
) -> list[SpeechSegment]:
    audio_path = Path(wav_path)
    pcm_audio, sample_rate = _read_wave(audio_path)
    frames = list(_generate_frames(FRAME_DURATION_MS, pcm_audio, sample_rate))
    if not frames:
        return []

    vad = webrtcvad.Vad(vad_mode)
    speech_intervals = _collect_speech_intervals(frames, sample_rate, vad, padding_ms=PADDING_MS)
    if not speech_intervals:
        return []

    merged_intervals = _merge_adjacent_intervals(speech_intervals, max_gap_seconds=MERGE_GAP_SECONDS)
    packed_intervals = _pack_intervals_into_clips(merged_intervals, min_clip_seconds, max_clip_seconds)
    if not packed_intervals:
        return []

    samples, loaded_sample_rate = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
    return _filter_segments(
        packed_intervals,
        samples,
        loaded_sample_rate,
        speech_intervals,
        min_clip_seconds,
        max_clip_seconds,
    )


def _read_wave(path: Path) -> tuple[bytes, int]:
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()

        if channels != 1:
            raise ValueError("VAD input must be mono.")
        if sample_width != 2:
            raise ValueError("VAD input must be 16-bit PCM.")
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"VAD input must be {TARGET_SAMPLE_RATE} Hz.")

        return wf.readframes(wf.getnframes()), sample_rate


def _generate_frames(frame_duration_ms: int, audio: bytes, sample_rate: int) -> Iterable[Frame]:
    bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    duration = bytes_per_frame / (sample_rate * 2.0)
    offset = 0
    timestamp = 0.0

    while offset + bytes_per_frame <= len(audio):
        yield Frame(audio[offset : offset + bytes_per_frame], timestamp, duration)
        timestamp += duration
        offset += bytes_per_frame


def _collect_speech_intervals(
    frames: list[Frame],
    sample_rate: int,
    vad: webrtcvad.Vad,
    padding_ms: int,
) -> list[tuple[float, float]]:
    if not frames:
        return []

    num_padding_frames = max(1, padding_ms // FRAME_DURATION_MS)
    ring_buffer: collections.deque[tuple[Frame, bool]] = collections.deque(maxlen=num_padding_frames)
    triggered = False
    intervals: list[tuple[float, float]] = []
    segment_start = 0.0

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes_data, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            voiced_count = sum(1 for _, voiced in ring_buffer if voiced)
            if voiced_count >= max(1, int(0.8 * ring_buffer.maxlen)):
                triggered = True
                segment_start = ring_buffer[0][0].timestamp
                ring_buffer.clear()
        else:
            ring_buffer.append((frame, is_speech))
            unvoiced_count = sum(1 for _, voiced in ring_buffer if not voiced)
            if unvoiced_count >= max(1, int(0.8 * ring_buffer.maxlen)):
                segment_end = frame.timestamp + frame.duration
                start = max(0.0, segment_start - (padding_ms / 1000.0))
                end = segment_end + (padding_ms / 1000.0)
                if end - start >= 0.45:
                    intervals.append((start, end))
                triggered = False
                ring_buffer.clear()

    if triggered:
        final_end = frames[-1].timestamp + frames[-1].duration + (padding_ms / 1000.0)
        start = max(0.0, segment_start - (padding_ms / 1000.0))
        if final_end - start >= 0.45:
            intervals.append((start, final_end))

    return intervals


def _merge_adjacent_intervals(
    intervals: list[tuple[float, float]],
    max_gap_seconds: float,
) -> list[tuple[float, float]]:
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap_seconds:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _pack_intervals_into_clips(
    intervals: list[tuple[float, float]],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[tuple[float, float]]:
    if not intervals:
        return []

    clips: list[tuple[float, float]] = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        current_duration = current_end - current_start
        proposed_duration = end - current_start
        gap = start - current_end

        if proposed_duration <= max_clip_seconds and (current_duration < min_clip_seconds or gap <= MERGE_GAP_SECONDS):
            current_end = end
        else:
            clips.extend(_split_long_interval(current_start, current_end, min_clip_seconds, max_clip_seconds))
            current_start, current_end = start, end

    clips.extend(_split_long_interval(current_start, current_end, min_clip_seconds, max_clip_seconds))
    return _rebalance_short_clips(clips, min_clip_seconds, max_clip_seconds)


def _split_long_interval(
    start: float,
    end: float,
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[tuple[float, float]]:
    duration = end - start
    if duration <= max_clip_seconds:
        return [(start, end)]

    clips: list[tuple[float, float]] = []
    cursor = start
    target_size = min(4.0, max_clip_seconds)

    while end - cursor > max_clip_seconds:
        next_end = min(end, cursor + target_size)
        if next_end - cursor < min_clip_seconds:
            next_end = min(end, cursor + min_clip_seconds)
        clips.append((cursor, next_end))
        cursor = next_end

    remaining = end - cursor
    if clips and remaining < min_clip_seconds:
        previous_start, _ = clips[-1]
        if end - previous_start <= max_clip_seconds:
            clips[-1] = (previous_start, end)
        elif remaining >= 0.75 * min_clip_seconds:
            clips.append((cursor, end))
    else:
        clips.append((cursor, end))

    return clips


def _rebalance_short_clips(
    clips: list[tuple[float, float]],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[tuple[float, float]]:
    if not clips:
        return []

    balanced: list[tuple[float, float]] = []
    for clip in clips:
        start, end = clip
        duration = end - start
        if balanced and duration < min_clip_seconds and end - balanced[-1][0] <= max_clip_seconds:
            previous_start, _ = balanced[-1]
            balanced[-1] = (previous_start, end)
        else:
            balanced.append(clip)

    if len(balanced) >= 2:
        last_start, last_end = balanced[-1]
        if last_end - last_start < min_clip_seconds:
            prev_start, prev_end = balanced[-2]
            if last_end - prev_start <= max_clip_seconds:
                balanced[-2] = (prev_start, last_end)
                balanced.pop()

    return [clip for clip in balanced if clip[1] - clip[0] >= min_clip_seconds]


def _filter_segments(
    candidate_segments: list[tuple[float, float]],
    samples: np.ndarray,
    sample_rate: int,
    speech_intervals: list[tuple[float, float]],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[SpeechSegment]:
    filtered_segments: list[SpeechSegment] = []

    for start, end in candidate_segments:
        duration = end - start
        if duration < min_clip_seconds or duration > max_clip_seconds:
            continue

        start_sample = max(0, int(start * sample_rate))
        end_sample = min(len(samples), int(end * sample_rate))
        clip_samples = samples[start_sample:end_sample]
        if clip_samples.size == 0:
            continue

        speech_ratio = _speech_overlap_ratio(start, end, speech_intervals)
        quality_score = _conversation_quality_score(clip_samples, sample_rate, speech_ratio)
        if speech_ratio < 0.35 or quality_score < 0.58:
            continue

        filtered_segments.append(
            SpeechSegment(
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                duration_seconds=round(duration, 3),
                speech_ratio=round(speech_ratio, 3),
                quality_score=round(quality_score, 3),
            )
        )

    return filtered_segments


def _speech_overlap_ratio(
    segment_start: float,
    segment_end: float,
    speech_intervals: list[tuple[float, float]],
) -> float:
    overlap = 0.0
    segment_duration = max(segment_end - segment_start, 1e-8)

    for speech_start, speech_end in speech_intervals:
        intersection = max(0.0, min(segment_end, speech_end) - max(segment_start, speech_start))
        overlap += intersection

    return min(1.0, overlap / segment_duration)


def _conversation_quality_score(samples: np.ndarray, sample_rate: int, speech_ratio: float) -> float:
    epsilon = 1e-8
    rms = librosa.feature.rms(y=samples).flatten()
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=samples)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=samples, sr=sample_rate)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=samples, sr=sample_rate)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=samples)))
    rms_db = float(np.mean(librosa.amplitude_to_db(np.maximum(rms, epsilon), ref=1.0)))

    harmonic, percussive = librosa.effects.hpss(samples)
    harmonic_energy = float(np.mean(np.abs(harmonic))) + epsilon
    percussive_energy = float(np.mean(np.abs(percussive))) + epsilon
    percussive_ratio = percussive_energy / harmonic_energy

    checks = [
        speech_ratio >= 0.40,
        rms_db > -38.0,
        flatness < 0.34,
        centroid < 2500.0,
        bandwidth < 3200.0,
        0.02 < zcr < 0.18,
        percussive_ratio < 1.15,
    ]

    return sum(checks) / len(checks)
