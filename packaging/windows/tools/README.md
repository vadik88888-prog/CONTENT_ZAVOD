# Bundled tools staging

`prepare_binaries.py` downloads and verifies the exact Windows x64
`ffmpeg.exe`, `ffprobe.exe`, and `yt-dlp.exe` versions pinned in
`../binaries.lock.json`. The binaries remain ignored by Git and are collected
only into local portable artifacts.
