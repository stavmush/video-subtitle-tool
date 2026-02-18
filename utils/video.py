"""
FFmpeg subtitle burning utilities.

Uses ffmpeg-python to burn SRT subtitles into a video file as hard subtitles
(permanently embedded, visible in any player without subtitle track support).

Hebrew / RTL text: libass handles Unicode BiDi automatically when the SRT file
contains Hebrew characters. The only requirement is a font with Hebrew glyph
coverage — provide its path via font_path.
"""

import ffmpeg
import os
import shutil


def burn_subtitles(
    input_video: str,
    srt_path: str,
    output_path: str,
    font_path: str | None = None,
    is_rtl: bool = False,
    font_size: int = 24,
) -> None:
    """
    Burn SRT subtitles into a video file.

    Args:
        input_video:  absolute path to the source MP4
        srt_path:     absolute path to the UTF-8 encoded .srt file
        output_path:  absolute path for the output MP4
        font_path:    path to a .ttf/.otf font file; required for correct Hebrew rendering
        is_rtl:       whether the subtitles are RTL (used for alignment hint)
        font_size:    subtitle font size in points

    Raises:
        RuntimeError:      if FFmpeg is not installed or exits with a non-zero code
        FileNotFoundError: if font_path is provided but the file does not exist
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found in PATH.\n"
            "Install it with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    if font_path and not os.path.isfile(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")

    vf_filter = _build_subtitles_filter(srt_path, font_path, font_size)

    try:
        (
            ffmpeg
            .input(input_video)
            .output(
                output_path,
                vf=vf_filter,
                acodec="copy",
                vcodec="libx264",
                crf=18,
                preset="fast",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed:\n{stderr}")


def _build_force_style(font_path: str | None, font_size: int) -> str:
    """Build the ASS force_style string for the FFmpeg subtitles filter."""
    parts = [
        f"FontSize={font_size}",
        "PrimaryColour=&H00FFFFFF&",   # white text  (ASS format: &H00BBGGRR&)
        "OutlineColour=&H00000000&",   # black outline
        "Outline=2",
        "Shadow=0",
        "MarginV=30",
        "Alignment=2",                 # centered, bottom
        "Bold=0",
    ]
    if font_path:
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        parts.append(f"FontName={font_name}")
    return ",".join(parts)


def _escape_filter_path(path: str) -> str:
    """Escape a file path for use inside an FFmpeg filtergraph string."""
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _build_subtitles_filter(srt_path: str, font_path: str | None, font_size: int) -> str:
    """
    Build the vf subtitles filter string.

    For Hebrew fonts, fontsdir points libass at the directory containing the
    font file; force_style/FontName selects it by name.
    """
    escaped_path = _escape_filter_path(srt_path)
    force_style = _build_force_style(font_path, font_size)

    parts = [f"subtitles='{escaped_path}'", f"force_style='{force_style}'"]

    if font_path:
        fonts_dir = _escape_filter_path(os.path.dirname(font_path))
        parts.append(f"fontsdir='{fonts_dir}'")

    return ":".join(parts)
