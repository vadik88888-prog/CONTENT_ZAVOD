# Bundled tools staging

`prepare_binaries.py` downloads and verifies the exact Windows x64
`ffmpeg.exe`, `ffprobe.exe`, `yt-dlp.exe`, and `deno.exe` versions pinned in
`../binaries.lock.json`. The binaries remain ignored by Git and are collected
only into local portable artifacts.

`prepare_youtube_access_runtime.py` stages the matching pinned BGutil
yt-dlp plugin and its Deno server under `../youtube-access-runtime/`. The
directory is a generated local build input, not user configuration.
