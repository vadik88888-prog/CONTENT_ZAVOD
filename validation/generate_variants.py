"""Create ignored synthetic source-format fixtures for the Goal 3E matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


PROFILES = {
    "720p30": ("1280:720", 30),
    "1080p30": ("1920:1080", 30),
    "1440p30": ("2560:1440", 30),
    "2160p30": ("3840:2160", 30),
    "720p24": ("1280:720", 24),
    "720p25": ("1280:720", 25),
    "720p50": ("1280:720", 50),
    "720p60": ("1280:720", 60),
    "vertical30": ("720:1280", 30),
    "square30": ("1080:1080", 30),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic technical source variants with FFmpeg.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), action="append")
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).parent / "fixtures")
    arguments = parser.parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to generate validation fixtures")
    source = arguments.source.resolve()
    if not source.is_file():
        raise SystemExit(f"Source fixture not found: {source}")
    profiles = arguments.profile or list(PROFILES)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    created: list[dict[str, object]] = []
    for name in profiles:
        resolution, fps = PROFILES[name]
        destination = arguments.output_directory / f"synthetic-format-{name}.mp4"
        filter_graph = f"scale={resolution}:force_original_aspect_ratio=decrease,pad={resolution}:(ow-iw)/2:(oh-ih)/2,fps={fps}"
        command = [
            ffmpeg, "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
            "-vf", filter_graph, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest", str(destination),
        ]
        subprocess.run(command, check=True)
        created.append({"profile": name, "path": str(destination.resolve()), "resolution": resolution, "fps": fps})
        print(destination)
    manifest = arguments.output_directory / "synthetic-format-manifest.json"
    manifest.write_text(json.dumps({"source": str(source), "profiles": created}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
