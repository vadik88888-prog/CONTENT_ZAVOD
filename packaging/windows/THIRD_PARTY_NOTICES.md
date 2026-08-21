# Third-party binary notices

The portable beta includes the exact binaries recorded in
`binaries.lock.json`.

- FFmpeg and FFprobe 8.0.1 are Gyan essentials builds obtained through the
  pinned `static_ffmpeg` source archive. Their reported configuration enables
  GPL and GPLv3 components. They are redistributed under GPL-3.0-or-later; see
  `licenses/LICENSE-ffmpeg-GPLv3.txt`. Corresponding upstream source is
  available from <https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz>; build
  provenance is recorded in the binary lock.
- yt-dlp 2026.07.04 is the official Windows x64 standalone release from the
  immutable GitHub release. It is distributed under the Unlicense; see
  `licenses/LICENSE-yt-dlp.txt`.
- Deno 2.9.5 is the official Windows x64 runtime from the immutable Deno GitHub
  release. It is bundled so yt-dlp can run the supported YouTube JavaScript
  challenge solver without browser cookies. Deno is distributed under the MIT
  License; see `licenses/LICENSE-deno-MIT.txt`.

The Python runtime and libraries collected by PyInstaller retain their own
package metadata and license files inside the onedir `_internal` tree where
provided by their distributions.
