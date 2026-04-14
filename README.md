---
title: Audio Dataset Labeler
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Hindi Audio Emotion Dataset Builder

This project extracts conversational Hindi speech clips from uploaded video files, presents them in a Gradio labeling interface, and saves accepted clips into emotion-specific dataset folders.

## Features

- Upload a local video file
- Extract mono 16 kHz WAV audio with FFmpeg
- Detect speech regions with `webrtcvad`
- Merge and split speech into 2-5 second clips
- Filter out silence, low-energy, and music-heavy segments
- Review clips in a Gradio UI with waveform playback
- Label clips into `happy`, `sad`, `angry`, `neutral`, `fear`, or `disgust`
- Undo the last label action
- Track labeled clips in `dataset/metadata.csv`

## Project Layout

```text
.
├── dataset/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── metadata.csv
├── src/
│   ├── audio_processing.py
│   ├── clip_manager.py
│   ├── ui.py
│   ├── utils.py
│   └── vad_segmentation.py
├── temp_clips/
├── app.py
├── main.py
├── README.md
└── requirements.txt
```

## Run The App

Hugging Face deploys this app automatically. To run locally:

```powershell
python main.py --host 127.0.0.1 --port 7860
```

Then open the local Gradio URL shown in the terminal.

## Labeling Workflow

1. Upload a video file.
2. Click **Extract Clips**.
3. Review the generated conversational clips in the scrollable list.
4. Listen to the active clip and choose an emotion label.
5. The clip is moved into `dataset/<emotion>/` and indexed in `dataset/metadata.csv`.
