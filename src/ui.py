from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr

from .audio_processing import AudioProcessingError, prepare_audio_source
from .clip_manager import ClipManager
from .utils import EMOTIONS, create_session_id, ensure_project_directories, format_seconds
from .vad_segmentation import segment_conversational_clips


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

:root {
  --bg: #0f1417;
  --surface: #1b2023;
  --surface-high: #262b2e;
  --surface-highest: #303539;
  --primary: #59d8de;
  --primary-glow: rgba(89, 216, 222, 0.3);
  --secondary: #76d6d5;
  --text: #dfe3e7;
  --text-muted: #bdc9c8;
  --error: #ffb4ab;
  --font-main: 'Inter', sans-serif;
  --font-display: 'Space Grotesk', sans-serif;
}

.gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-main);
  max-width: 1400px !important;
}

h1, h2, h3 {
  font-family: var(--font-display);
  font-weight: 700;
  color: var(--primary) !important;
  letter-spacing: -0.02em;
}

.panel-card {
  background: var(--surface) !important;
  border: 1px solid rgba(89, 216, 222, 0.1) !important;
  border-radius: 12px !important;
  padding: 20px !important;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
}

.session-note {
  background: rgba(89, 216, 222, 0.05) !important;
  border: 1px solid rgba(89, 216, 222, 0.15) !important;
  border-radius: 8px !important;
  padding: 12px 16px !important;
  font-family: var(--font-main);
}

#clip-list {
  min-height: 500px;
  max-height: 500px;
  overflow-y: auto;
  padding: 8px;
  border-radius: 8px;
  background: var(--surface-high);
}

.clip-row {
  display: grid;
  grid-template-columns: 48px 1fr auto;
  gap: 16px;
  align-items: center;
  padding: 14px 18px;
  border-radius: 8px;
  margin-bottom: 8px;
  background: var(--surface);
  border: 1px solid transparent;
  transition: all 0.2s ease;
  cursor: pointer;
}

.clip-row:hover {
  background: var(--surface-highest);
}

.clip-row.current {
  border-color: var(--primary);
  background: var(--surface-highest);
  box-shadow: 0 0 15px var(--primary-glow);
}

.clip-row .index {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--primary);
}

.clip-row .meta {
  font-size: 0.9rem;
  color: var(--text);
  line-height: 1.5;
}

.clip-row .status {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: 4px;
  background: var(--surface-highest);
  color: var(--text-muted);
  letter-spacing: 0.05em;
}

.clip-row .status.labeled {
  background: rgba(89, 216, 222, 0.15);
  color: var(--primary);
}

.clip-row .status.skipped {
  background: rgba(255, 180, 171, 0.15);
  color: var(--error);
}

/* Emotion Chips */
.emotion-btn {
  background: var(--surface-high) !important;
  border: 1px solid rgba(89, 216, 222, 0.2) !important;
  transition: all 0.2s ease !important;
}

.emotion-btn:hover {
  background: var(--surface-highest) !important;
  border-color: var(--primary) !important;
  box-shadow: 0 0 10px var(--primary-glow) !important;
}

/* Audio Player Stylings */
audio {
  filter: invert(80%) hue-rotate(150deg) brightness(1.2);
}
"""


clip_manager = ClipManager()


def build_demo() -> gr.Blocks:
    ensure_project_directories()

    with gr.Blocks(css=CUSTOM_CSS, title="Hindi Audio Emotion Dataset Builder") as demo:
        app_state = gr.State(_empty_state())

        gr.Markdown(
            """
            # Hindi Audio Emotion Dataset Builder
            Upload a video file, generate 2-5 second speech clips, then label each clip into the dataset buckets.
            """
        )

        with gr.Row(elem_classes=["panel-card"]):
            with gr.Column(scale=9):
                video_input = gr.File(
                    label="Upload Video Source",
                    file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"],
                    type="filepath",
                )
            with gr.Column(scale=3):
                build_button = gr.Button("🚀 Extract Clips from Source", variant="primary", scale=1)
                notice = gr.Markdown(value="*Status:* Ready for a new source.")

        session_summary = gr.Markdown(value=_render_session_summary(_empty_state()), elem_classes=["session-note"])

        with gr.Row():
            with gr.Column(scale=6, elem_classes=["panel-card"]):
                gr.Markdown("### Extracted Clips")
                clip_list = gr.HTML(value=_render_clip_list(_empty_state()), elem_id="clip-list")
            with gr.Column(scale=6, elem_classes=["panel-card"]):
                clip_counter = gr.Markdown(value=_render_clip_counter(_empty_state()))
                clip_details = gr.Markdown(value=_render_clip_details(_empty_state()))
                audio_player = gr.Audio(
                    value=None,
                    type="filepath",
                    label="Current Clip",
                    interactive=False,
                    autoplay=False,
                )

                with gr.Row():
                    previous_button = gr.Button("Previous")
                    next_button = gr.Button("Next")
                    skip_button = gr.Button("Skip")
                    undo_button = gr.Button("Undo Label")

                emotion_buttons: dict[str, gr.Button] = {}
                for emotion_group in (EMOTIONS[:3], EMOTIONS[3:]):
                    with gr.Row():
                        for emotion in emotion_group:
                            emotion_buttons[emotion] = gr.Button(
                                emotion.title(), 
                                variant="secondary", 
                                elem_classes=["emotion-btn"]
                            )

        outputs = [
            app_state,
            audio_player,
            clip_counter,
            clip_details,
            clip_list,
            session_summary,
            notice,
        ]

        build_button.click(
            fn=process_source,
            inputs=[video_input, app_state],
            outputs=outputs,
            show_progress="minimal",
        )

        previous_button.click(
            fn=go_previous,
            inputs=[app_state],
            outputs=outputs,
            show_progress="hidden",
            queue=False,
        )
        next_button.click(
            fn=go_next,
            inputs=[app_state],
            outputs=outputs,
            show_progress="hidden",
            queue=False,
        )
        skip_button.click(
            fn=skip_current_clip,
            inputs=[app_state],
            outputs=outputs,
            show_progress="hidden",
            queue=False,
        )
        undo_button.click(
            fn=undo_last_label,
            inputs=[app_state],
            outputs=outputs,
            show_progress="hidden",
            queue=False,
        )

        for emotion, button in emotion_buttons.items():
            button.click(
                fn=lambda state, emotion_name=emotion: label_current_clip(state, emotion_name),
                inputs=[app_state],
                outputs=outputs,
                show_progress="hidden",
                queue=False,
            )

    return demo


def process_source(
    video_file: str | None,
    _state: dict[str, Any],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    state = _empty_state()
    try:
        session_id = create_session_id()
        progress(0.1, desc="Preparing source")
        prepared = prepare_audio_source(video_file, session_id)

        progress(0.5, desc="Running speech segmentation")
        segments = segment_conversational_clips(prepared.audio_path)
        if not segments:
            return _compose_view(
                state,
                "No conversational clips were detected. Try a clearer speech-heavy source.",
            )

        progress(0.8, desc="Saving clips")
        clip_records = clip_manager.create_clips_from_segments(
            source_audio_path=prepared.audio_path,
            source_video=prepared.source_reference,
            session_id=session_id,
            segments=segments,
        )

        state.update(
            {
                "session_id": session_id,
                "source_video": prepared.source_reference,
                "source_label": prepared.source_label,
                "audio_path": prepared.audio_path,
                "clips": [record.to_dict() for record in clip_records],
                "current_index": 0,
                "history": [],
            }
        )
        return _compose_view(state, f"Prepared {len(clip_records)} clips from {prepared.source_label}.")
    except (AudioProcessingError, ValueError) as exc:
        return _compose_view(state, f"Unable to process source: {exc}")
    except Exception as exc:  # pragma: no cover - Gradio runtime surface
        return _compose_view(state, f"Unexpected error: {exc}")


def go_previous(state: dict[str, Any]):
    if not state["clips"]:
        return _compose_view(state, "Load a source to begin labeling.")

    state["current_index"] = max(0, state["current_index"] - 1)
    return _compose_view(state, "Moved to the previous clip.")


def go_next(state: dict[str, Any]):
    if not state["clips"]:
        return _compose_view(state, "Load a source to begin labeling.")

    state["current_index"] = min(len(state["clips"]) - 1, state["current_index"] + 1)
    return _compose_view(state, "Moved to the next clip.")


def skip_current_clip(state: dict[str, Any]):
    clip = _current_clip(state)
    if clip is None:
        return _compose_view(state, "No clip is currently selected.")

    if clip["status"] == "labeled":
        return _compose_view(state, "This clip is already labeled. Use an emotion button to relabel it if needed.")

    clip_manager.mark_skipped(clip)
    state["clips"][state["current_index"]] = clip
    state["current_index"] = _next_focus_index(state["clips"], state["current_index"])
    return _compose_view(state, f"Skipped clip {clip['clip_id']}.")


def label_current_clip(state: dict[str, Any], label: str):
    clip = _current_clip(state)
    if clip is None:
        return _compose_view(state, "No clip is currently selected.")
    if clip["label"] == label and clip["status"] == "labeled":
        return _compose_view(state, f"Clip {clip['clip_id']} is already labeled as {label}.")

    history_entry = {
        "clip_id": clip["clip_id"],
        "index": state["current_index"],
        "previous_path": clip["clip_path"],
        "previous_status": clip["status"],
        "previous_label": clip["label"],
        "previous_filename": clip["filename"],
    }
    try:
        clip_manager.label_clip(clip, label)
    except Exception as exc:  # pragma: no cover - filesystem/runtime surface
        return _compose_view(state, f"Could not label clip {clip['clip_id']}: {exc}")

    state["clips"][state["current_index"]] = clip
    state["history"].append(history_entry)
    state["current_index"] = _next_focus_index(state["clips"], state["current_index"])

    complete = all(item["status"] == "labeled" for item in state["clips"])
    message = (
        f"Labeled clip {clip['clip_id']} as {label}."
        if not complete
        else "All clips in this session are labeled."
    )
    return _compose_view(state, message)


def undo_last_label(state: dict[str, Any]):
    if not state["history"]:
        return _compose_view(state, "There is no labeling action to undo.")

    last_action = state["history"].pop()
    clip = next((item for item in state["clips"] if item["clip_id"] == last_action["clip_id"]), None)
    if clip is None:
        return _compose_view(state, "The clip for the last action could not be found.")

    try:
        restored_clip = clip_manager.undo_label(
            clip=clip,
            previous_path=last_action["previous_path"],
            previous_status=last_action["previous_status"],
            previous_label=last_action["previous_label"],
            previous_filename=last_action["previous_filename"],
        )
    except Exception as exc:  # pragma: no cover - filesystem/runtime surface
        return _compose_view(state, f"Could not undo the last label: {exc}")

    state["current_index"] = last_action["index"]
    state["clips"][state["current_index"]] = restored_clip
    return _compose_view(state, f"Undid the last label for clip {restored_clip['clip_id']}.")


def _empty_state() -> dict[str, Any]:
    return {
        "session_id": "",
        "source_video": "",
        "source_label": "",
        "audio_path": "",
        "clips": [],
        "current_index": 0,
        "history": [],
    }


def _current_clip(state: dict[str, Any]) -> dict[str, Any] | None:
    clips = state.get("clips", [])
    if not clips:
        return None
    index = max(0, min(state.get("current_index", 0), len(clips) - 1))
    state["current_index"] = index
    return clips[index]


def _next_focus_index(clips: list[dict[str, Any]], current_index: int) -> int:
    if not clips:
        return 0

    for index in range(current_index + 1, len(clips)):
        if clips[index]["status"] != "labeled":
            return index

    for index, clip in enumerate(clips):
        if clip["status"] != "labeled":
            return index

    return min(current_index, len(clips) - 1)


def _compose_view(state: dict[str, Any], notice_message: str):
    current_clip = _current_clip(state)
    audio_value = None
    if current_clip is not None:
        clip_path = Path(current_clip["clip_path"])
        audio_value = str(clip_path) if clip_path.exists() else None

    return (
        state,
        audio_value,
        _render_clip_counter(state),
        _render_clip_details(state),
        _render_clip_list(state),
        _render_session_summary(state),
        notice_message,
    )


def _render_clip_counter(state: dict[str, Any]) -> str:
    total = len(state["clips"])
    if total == 0:
        return "### Current Clip\nNo clips loaded yet."

    current_position = state["current_index"] + 1
    return f"### Current Clip\nClip {current_position} of {total}"


def _render_clip_details(state: dict[str, Any]) -> str:
    clip = _current_clip(state)
    if clip is None:
        return "No clip selected."

    label_text = clip["label"].title() if clip["label"] else "Unlabeled"
    status_text = clip["status"].title()
    duration = format_seconds(clip["duration_seconds"])
    return (
        f"**ID:** `{clip['clip_id']}`  \n"
        f"**Time:** {clip['timestamp']}  \n"
        f"**Duration:** {duration}  \n"
        f"**Status:** {status_text}  \n"
        f"**Label:** {label_text}"
    )


def _render_session_summary(state: dict[str, Any]) -> str:
    clips = state["clips"]
    total = len(clips)
    labeled = sum(1 for clip in clips if clip["status"] == "labeled")
    skipped = sum(1 for clip in clips if clip["status"] == "skipped")
    unlabeled = total - labeled - skipped

    if total == 0:
        return (
            "**Session:** idle  \n"
            "**Source:** none  \n"
            "**Counts:** 0 total, 0 labeled, 0 skipped, 0 pending"
        )

    source_value = html.escape(state["source_video"])
    return (
        f"**Session:** `{state['session_id']}`  \n"
        f"**Source:** {source_value}  \n"
        f"**Counts:** {total} total, {labeled} labeled, {skipped} skipped, {unlabeled} pending"
    )


def _render_clip_list(state: dict[str, Any]) -> str:
    if not state["clips"]:
        return "<div class='clip-row'><div class='meta'>No clips generated yet.</div></div>"

    rows = []
    for index, clip in enumerate(state["clips"], start=1):
        status = clip["status"]
        label = clip["label"].title() if clip["label"] else status.title()
        css_class = "clip-row current" if index - 1 == state["current_index"] else "clip-row"
        rows.append(
            (
                f"<div class='{css_class}'>"
                f"<div class='index'>#{index}</div>"
                f"<div class='meta'><strong>{html.escape(clip['timestamp'])}</strong><br>"
                f"{format_seconds(clip['duration_seconds'])} | {html.escape(clip['clip_id'])}</div>"
                f"<div class='status {html.escape(status)}'>{html.escape(label)}</div>"
                "</div>"
            )
        )

    return "".join(rows)
