"""
SRT utility functions: convert between Whisper segments, pandas DataFrames, and SRT strings.
"""

import srt
import pandas as pd
from datetime import timedelta
from typing import Any

Segment = dict[str, Any]

SUBTITLE_COLUMNS = ["index", "start", "end", "text"]


def _seconds_to_timedelta(seconds: float) -> timedelta:
    return timedelta(seconds=seconds)


def _srt_timestamp_to_str(td: timedelta) -> str:
    """Convert timedelta to SRT timestamp string: HH:MM:SS,mmm"""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _str_to_timedelta(ts: str) -> timedelta:
    """
    Parse SRT timestamp string to timedelta.
    Accepts: "HH:MM:SS,mmm"

    Raises:
        ValueError: on malformed input
    """
    try:
        time_part, ms_part = ts.strip().split(",")
        h, m, s = time_part.split(":")
        return timedelta(
            hours=int(h),
            minutes=int(m),
            seconds=int(s),
            milliseconds=int(ms_part),
        )
    except Exception:
        raise ValueError(f"Invalid SRT timestamp: '{ts}'")


def segments_to_dataframe(segments: list[Segment]) -> pd.DataFrame:
    """Convert Whisper-style segments to a pandas DataFrame for st.data_editor."""
    rows = [
        {
            "index": seg["id"],
            "start": _srt_timestamp_to_str(_seconds_to_timedelta(seg["start"])),
            "end":   _srt_timestamp_to_str(_seconds_to_timedelta(seg["end"])),
            "text":  seg["text"],
        }
        for seg in segments
    ]
    return pd.DataFrame(rows, columns=SUBTITLE_COLUMNS)


def dataframe_to_srt(df: pd.DataFrame) -> str:
    """
    Convert an edited DataFrame back to an SRT-formatted string.

    Validates that start < end for every row.

    Raises:
        ValueError: if any timestamp is invalid or start >= end

    Returns:
        UTF-8 string in SRT format ready to write to a .srt file
    """
    subtitles = []
    df = df.reset_index(drop=True)
    for _, row in df.iterrows():
        start = _str_to_timedelta(str(row["start"]))
        end   = _str_to_timedelta(str(row["end"]))
        if start >= end:
            raise ValueError(
                f"Subtitle #{row['index']}: start ({row['start']}) >= end ({row['end']})"
            )
        subtitles.append(
            srt.Subtitle(
                index=int(row["index"]),
                start=start,
                end=end,
                content=str(row["text"]),
            )
        )
    return srt.compose(subtitles)


def parse_srt_to_dataframe(srt_text: str) -> pd.DataFrame:
    """Parse an SRT string into a DataFrame (e.g. for loading existing subtitles)."""
    parsed = list(srt.parse(srt_text))
    rows = [
        {
            "index": sub.index,
            "start": _srt_timestamp_to_str(sub.start),
            "end":   _srt_timestamp_to_str(sub.end),
            "text":  sub.content,
        }
        for sub in parsed
    ]
    return pd.DataFrame(rows, columns=SUBTITLE_COLUMNS)


def merge_srt_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate SRT DataFrames, offsetting each by the end time of the previous."""
    merged = []
    offset = timedelta(0)
    for df in dfs:
        shifted = df.copy()
        shifted["start"] = shifted["start"].apply(
            lambda t: _srt_timestamp_to_str(_str_to_timedelta(t) + offset)
        )
        shifted["end"] = shifted["end"].apply(
            lambda t: _srt_timestamp_to_str(_str_to_timedelta(t) + offset)
        )
        offset += _str_to_timedelta(df["end"].iloc[-1])
        merged.append(shifted)
    result = pd.concat(merged, ignore_index=True)
    result["index"] = range(1, len(result) + 1)
    return result
