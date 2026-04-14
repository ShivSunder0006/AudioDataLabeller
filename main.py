from __future__ import annotations

import argparse
from pathlib import Path

from src.ui import build_demo
from src.utils import DATASET_DIR, TEMP_CLIPS_DIR, ensure_project_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hindi Audio Emotion Dataset Builder")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the Gradio app.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio app.")
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio share link.")
    parser.add_argument("--debug", action="store_true", help="Launch Gradio in debug mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_directories()

    demo = build_demo()
    demo.queue(api_open=False).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        allowed_paths=[
            str(Path(DATASET_DIR).resolve()),
            str(Path(TEMP_CLIPS_DIR).resolve()),
        ],
    )


if __name__ == "__main__":
    main()
