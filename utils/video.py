"""
FFmpeg subtitle utilities: embed (soft) or burn (hard) subtitles into MP4.

Soft embedding  (always available):
  Adds the subtitle track to the MP4 container. The subtitles are selectable
  in any player (VLC, IINA, QuickTime). Fast — no video re-encoding needed.

Hard burning    (requires FFmpeg compiled with --enable-libass):
  Renders subtitle text permanently onto each video frame. Visible in any
  player even without subtitle support. Requires libass.
  Install: brew tap homebrew-ffmpeg/ffmpeg &&
           brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass
"""

import ffmpeg
import os
import shutil
import subprocess


def _has_libass() -> bool:
    """Return True if the system ffmpeg was compiled with libass."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return "--enable-libass" in result.stdout
    except FileNotFoundError:
        return False


_LANG_ISO639_2 = {"en": "eng", "he": "heb"}
_LANG_TITLE = {"en": "English", "he": "Hebrew"}


def embed_subtitles_multi(
    input_video: str,
    srt_tracks: list[tuple[str, str]],  # list of (srt_path, lang_code)
    output_path: str,
) -> None:
    """
    Embed multiple SRT subtitle tracks into an MP4 container.

    Each track is selectable independently in VLC, IINA, QuickTime, etc.
    No re-encoding — video and audio are stream-copied.

    Args:
        srt_tracks: list of (srt_path, lang_code) where lang_code is 'en' or 'he'

    Raises:
        RuntimeError: if FFmpeg fails
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")

    cmd = ["ffmpeg", "-y", "-i", input_video]
    for srt_path, _ in srt_tracks:
        cmd += ["-i", srt_path]

    cmd += ["-map", "0:v", "-map", "0:a"]
    for i in range(len(srt_tracks)):
        cmd += ["-map", str(i + 1)]

    cmd += ["-c:v", "copy", "-c:a", "copy"]
    for i in range(len(srt_tracks)):
        cmd += [f"-c:s:{i}", "mov_text"]

    for i, (_, lang) in enumerate(srt_tracks):
        iso = _LANG_ISO639_2.get(lang, lang)
        title = _LANG_TITLE.get(lang, lang)
        cmd += [f"-metadata:s:s:{i}", f"language={iso}", f"-metadata:s:s:{i}", f"title={title}"]

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr.decode('utf-8', errors='replace')}")


def embed_subtitles(
    input_video: str,
    srt_path: str,
    output_path: str,
) -> None:
    """
    Embed SRT subtitles as a soft track inside the MP4 container (mov_text codec).

    Fast — video and audio are stream-copied (no re-encoding).
    The subtitle track is selectable in VLC, IINA, QuickTime, etc.

    Raises:
        RuntimeError: if FFmpeg fails
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")

    try:
        (
            ffmpeg
            .output(
                ffmpeg.input(input_video),
                ffmpeg.input(srt_path),
                output_path,
                **{"c:v": "copy", "c:a": "copy", "c:s": "mov_text"},
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg failed:\n{e.stderr.decode('utf-8', errors='replace')}")


def burn_subtitles(
    input_video: str,
    srt_path: str,
    output_path: str,
    font_path: str | None = None,
    is_rtl: bool = False,
    font_size: int = 24,
) -> None:
    """
    Burn subtitles permanently into video frames (hard subtitles).
    Requires FFmpeg compiled with --enable-libass.

    Raises:
        RuntimeError:      if FFmpeg lacks libass or fails
        FileNotFoundError: if font_path is provided but doesn't exist
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")

    if not _has_libass():
        raise RuntimeError(
            "Hard subtitle burning requires FFmpeg with libass support.\n"
            "Your current FFmpeg build does not include libass.\n\n"
            "To enable it:\n"
            "  brew tap homebrew-ffmpeg/ffmpeg\n"
            "  brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass\n\n"
            "For now, use 'Embed subtitles' instead — it works with any player (VLC, IINA, etc)."
        )

    if font_path and not os.path.isfile(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")

    # Copy font with a simple name — variable font filenames like
    # NotoSansHebrew[wdth,wght].ttf contain [ ] which break FFmpeg's filtergraph parser
    simple_font = None
    if font_path:
        ext = os.path.splitext(font_path)[1]
        simple_font = os.path.join(os.path.dirname(srt_path), f"subtitle_font{ext}")
        shutil.copy2(font_path, simple_font)

    force_style = ",".join([
        f"FontSize={font_size}",
        "PrimaryColour=&H00FFFFFF&",
        "OutlineColour=&H00000000&",
        "Outline=2", "Shadow=0", "MarginV=30", "Alignment=2", "Bold=0",
    ])

    def esc(p: str) -> str:
        return p.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")

    parts = [f"subtitles='{esc(srt_path)}'", f"force_style='{force_style}'"]
    if simple_font:
        parts.append(f"fontsdir='{esc(os.path.dirname(simple_font))}'")
    vf = ":".join(parts)

    try:
        (
            ffmpeg
            .input(input_video)
            .output(output_path, vf=vf, acodec="copy", vcodec="libx264", crf=18, preset="fast")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg failed:\n{e.stderr.decode('utf-8', errors='replace')}")
    finally:
        if simple_font and os.path.exists(simple_font):
            os.remove(simple_font)
