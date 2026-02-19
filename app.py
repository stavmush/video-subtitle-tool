"""
Video Subtitle Tool — Streamlit app

5-step pipeline:
  1. Upload MP4
  2. Transcribe with Whisper
  3. Translate to English or Hebrew (offline, Helsinki-NLP)
  4. Edit subtitles in a table
  5. Export as SRT and/or burned-in MP4
"""

import os
import shutil
import tempfile

import pandas as pd
import streamlit as st

from utils.srt_utils import dataframe_to_srt, segments_to_dataframe
from utils.transcribe import transcribe_to_english, transcribe_video
from utils.translate import translate_segments
from utils.video import burn_subtitles, embed_subtitles

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Video Subtitle Tool", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────
STATE_DEFAULTS: dict = {
    "uploaded_video_path": None,
    "subtitles_df": None,
    "whisper_segments": None,
    "source_language": None,
    "transcription_done": False,
    "translation_done": False,
    "srt_content": None,
    "temp_dir": None,
}

for key, default in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _cleanup_temp(tmp_dir: str | None) -> None:
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")

    # Memory figures are for faster-whisper with int8 quantization (~4x less than openai-whisper)
    MODEL_INFO = {
        "tiny":        "~40 MB  · fastest · low accuracy",
        "base":        "~75 MB  · fast    · decent accuracy",
        "small":       "~240 MB · good balance (recommended)",
        "medium":      "~770 MB · high accuracy · fine on 8 GB RAM",
        "large-v2":    "~1.5 GB · best accuracy · needs 4 GB+ free RAM",
        "large-v3":    "~1.5 GB · latest model  · needs 4 GB+ free RAM",
    }

    whisper_model_size = st.selectbox(
        "Whisper model size",
        list(MODEL_INFO.keys()),
        index=2,
        format_func=lambda m: f"{m}  —  {MODEL_INFO[m]}",
    )

    if whisper_model_size in ("large-v2", "large-v3"):
        st.warning(
            "The **large** model needs ~1.5 GB RAM. On an 8 GB machine, "
            "**close Chrome, Slack, and other heavy apps** before transcribing "
            "or the OS will kill the process mid-run."
        )

    target_language = st.radio(
        "Subtitle language",
        ["en", "he"],
        format_func=lambda x: "English" if x == "en" else "Hebrew (עברית)",
    )

    st.divider()

    if st.button("Reset / Start Over", type="secondary", use_container_width=True):
        _cleanup_temp(st.session_state["temp_dir"])
        for key, default in STATE_DEFAULTS.items():
            st.session_state[key] = default
        st.rerun()

    st.caption("Models are downloaded on first use and cached locally.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Video Subtitle Tool")
st.caption("Transcribe, translate, edit, and export subtitles for your videos.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ═══════════════════════════════════════════════════════════════════════════════
st.header("1. Upload Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mkv", "avi", "mov"],
    label_visibility="collapsed",
)

if uploaded_file and st.session_state["uploaded_video_path"] is None:
    tmp_dir = tempfile.mkdtemp(prefix="subtitle_tool_")
    st.session_state["temp_dir"] = tmp_dir
    video_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["uploaded_video_path"] = video_path

if st.session_state["uploaded_video_path"]:
    video_path = st.session_state["uploaded_video_path"]
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb <= 200:
        st.video(video_path)
    else:
        st.info(
            f"Video uploaded: **{os.path.basename(video_path)}** "
            f"({file_size_mb:.0f} MB) — preview skipped for large files to save RAM."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Transcribe
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["uploaded_video_path"]:
    st.header("2. Transcribe Audio")

    if not st.session_state["transcription_done"]:
        if st.button("Transcribe", type="primary"):
            status_text = st.empty()
            progress_bar = st.progress(0)

            is_first_load = not any(
                True for _ in []  # placeholder; model cache checked below
            )
            status_text.info(
                f"Loading Whisper **{whisper_model_size}** model into memory "
                f"(first load may take a minute for larger models)..."
            )

            def on_transcribe_progress(fraction: float):
                progress_bar.progress(fraction)
                status_text.caption(f"Transcribing audio... {int(fraction * 100)}%")

            try:
                segments, detected_lang = transcribe_video(
                    video_path=st.session_state["uploaded_video_path"],
                    model_size=whisper_model_size,
                    on_progress=on_transcribe_progress,
                )
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()
                st.session_state["whisper_segments"] = segments
                st.session_state["source_language"] = detected_lang
                st.session_state["transcription_done"] = True
                st.session_state["subtitles_df"] = segments_to_dataframe(segments)
                st.rerun()
            except MemoryError:
                progress_bar.empty()
                status_text.empty()
                st.error(
                    f"Out of memory loading the **{whisper_model_size}** model. "
                    "Try a smaller model size (medium or small) in the sidebar."
                )
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Transcription failed: {e}")
    else:
        src = st.session_state["source_language"]
        st.success(f"Transcription complete. Detected language: **{src}**")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Translate
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["transcription_done"]:
    st.header("3. Translate")

    src = st.session_state["source_language"]
    tgt = target_language

    already_target = (tgt == "en" and src == "en") or (tgt == "he" and src == "he")

    if already_target:
        st.info("Source language matches target — no translation needed.")
        st.session_state["translation_done"] = True

    elif not st.session_state["translation_done"]:
        lang_label = "Hebrew (עברית)" if tgt == "he" else "English"
        if st.button(f"Translate to {lang_label}", type="primary"):
            try:
                if tgt == "he" and src != "en":
                    # Stage 1: Whisper translate to English
                    t_status = st.empty()
                    t_bar = st.progress(0)
                    t_status.caption("Stage 1/2: Translating to English with Whisper...")

                    def on_translate_progress(f):
                        t_bar.progress(f)
                        t_status.caption(f"Stage 1/2: Translating to English... {int(f*100)}%")

                    english_segs = transcribe_to_english(
                        video_path=st.session_state["uploaded_video_path"],
                        model_size=whisper_model_size,
                        on_progress=on_translate_progress,
                    )
                    t_bar.empty()
                    t_status.empty()

                    # Stage 2: English → Hebrew with MarianMT
                    with st.spinner(
                        "Stage 2/2: Translating English → Hebrew (downloading model on first run)..."
                    ):
                        translated = translate_segments(
                            segments=st.session_state["whisper_segments"],
                            source_lang=src,
                            target_lang="he",
                            whisper_segments_english=english_segs,
                        )

                elif tgt == "en" and src != "en":
                    # Whisper's built-in translate task
                    t_status = st.empty()
                    t_bar = st.progress(0)
                    t_status.caption("Translating to English with Whisper...")

                    def on_translate_progress(f):
                        t_bar.progress(f)
                        t_status.caption(f"Translating to English... {int(f*100)}%")

                    translated = transcribe_to_english(
                        video_path=st.session_state["uploaded_video_path"],
                        model_size=whisper_model_size,
                        on_progress=on_translate_progress,
                    )
                    t_bar.empty()
                    t_status.empty()

                else:
                    # src == "en", tgt == "he"
                    with st.spinner(
                        "Translating English → Hebrew (downloading model on first run)..."
                    ):
                        translated = translate_segments(
                            segments=st.session_state["whisper_segments"],
                            source_lang=src,
                            target_lang="he",
                        )

                st.session_state["subtitles_df"] = segments_to_dataframe(translated)
                st.session_state["translation_done"] = True
                st.rerun()

            except Exception as e:
                st.error(f"Translation failed: {e}")

    else:
        lang_label = "Hebrew (עברית)" if tgt == "he" else "English"
        st.success(f"Translation to {lang_label} complete.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Edit Subtitles
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["translation_done"]:
    st.header("4. Edit Subtitles")
    st.info(
        "**Click any cell to edit it.** "
        "You can change the subtitle text, adjust start/end timestamps (format: HH:MM:SS,mmm), "
        "or use the row controls on the left to delete a row. "
        "Use the **Renumber** button if you delete rows to keep numbering clean."
    )

    # RTL CSS injection for Hebrew
    if target_language == "he":
        st.markdown(
            """
            <style>
            [data-testid="stDataFrameResizable"] input[type="text"] {
                direction: rtl;
                text-align: right;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Hebrew text is RTL. The editor may display it LTR — "
            "this is a browser limitation. The exported SRT and burned video will render correctly."
        )

    df = st.session_state["subtitles_df"]

    col_left, col_right = st.columns([4, 1])
    with col_right:
        if st.button("Renumber subtitles"):
            df = df.copy()
            df["index"] = range(1, len(df) + 1)
            st.session_state["subtitles_df"] = df
            st.rerun()

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "index": st.column_config.NumberColumn(
                "No.",
                min_value=1,
                step=1,
                width="small",
            ),
            "start": st.column_config.TextColumn(
                "Start",
                width="medium",
                help="Format: HH:MM:SS,mmm",
                validate=r"^\d{2}:\d{2}:\d{2},\d{3}$",
            ),
            "end": st.column_config.TextColumn(
                "End",
                width="medium",
                help="Format: HH:MM:SS,mmm",
                validate=r"^\d{2}:\d{2}:\d{2},\d{3}$",
            ),
            "text": st.column_config.TextColumn(
                "Subtitle Text",
                width="large",
            ),
        },
        hide_index=True,
        key="subtitle_editor",
    )

    # Persist edits and regenerate SRT
    st.session_state["subtitles_df"] = edited_df
    try:
        srt_text = dataframe_to_srt(edited_df)
        st.session_state["srt_content"] = srt_text
    except ValueError as e:
        st.warning(f"SRT validation issue: {e}")
        st.session_state["srt_content"] = None

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Export
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["translation_done"]:
    st.header("5. Export")

    col1, col2 = st.columns(2)

    # ── 5a: Download SRT ──────────────────────────────────────────────────────
    with col1:
        st.subheader("Download SRT file")
        st.caption("Import this .srt file into any video player (VLC, IINA, etc.).")
        if st.session_state["srt_content"]:
            st.download_button(
                label="Download .srt",
                data=st.session_state["srt_content"].encode("utf-8"),
                file_name="subtitles.srt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.warning("Fix timestamp errors above before downloading.")

    # ── 5b: Embed / Burn subtitles into video ────────────────────────────────
    with col2:
        st.subheader("Add subtitles to video")

        embed_tab, burn_tab = st.tabs(["Embed (Recommended)", "Burn (Hard subs)"])

        # ── Embed (soft track) ────────────────────────────────────────────────
        with embed_tab:
            st.caption(
                "Adds a subtitle track inside the MP4 container — **no re-encoding**. "
                "Selectable in VLC, IINA, QuickTime, and most players. Fast."
            )
            if st.button("Embed subtitles", type="primary", use_container_width=True):
                if not st.session_state["srt_content"]:
                    st.error("Fix subtitle errors before embedding.")
                else:
                    tmp_dir = st.session_state["temp_dir"]
                    srt_path = os.path.join(tmp_dir, "subtitles.srt")
                    output_path = os.path.join(tmp_dir, "output_embedded.mp4")

                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state["srt_content"])

                    with st.spinner("Embedding subtitles with FFmpeg..."):
                        try:
                            embed_subtitles(
                                input_video=st.session_state["uploaded_video_path"],
                                srt_path=srt_path,
                                output_path=output_path,
                            )

                            st.success("Done! Click below to download.")
                            with open(output_path, "rb") as vf:
                                video_bytes = vf.read()
                            st.download_button(
                                label="Download video with embedded subtitles",
                                data=video_bytes,
                                file_name="video_with_subtitles.mp4",
                                mime="video/mp4",
                                use_container_width=True,
                            )
                        except (RuntimeError, FileNotFoundError) as e:
                            st.error(str(e))

        # ── Burn (hard subs, requires libass) ─────────────────────────────────
        with burn_tab:
            st.caption(
                "Renders subtitles permanently onto each video frame. "
                "Visible in **any** player, even without subtitle support. "
                "Requires FFmpeg compiled with libass."
            )

            if target_language == "he":
                font_path = st.text_input(
                    "Hebrew font path (TTF/OTF with Hebrew glyphs)",
                    value="/Users/stav/Library/Fonts/NotoSansHebrew[wdth,wght].ttf",
                    help=(
                        "Required for correct Hebrew rendering. "
                        "Font installed at: /Users/stav/Library/Fonts/NotoSansHebrew[wdth,wght].ttf"
                    ),
                )
            else:
                font_path = None

            font_size = st.slider("Font size", min_value=14, max_value=48, value=24, step=2)

            if st.button("Burn subtitles", type="primary", use_container_width=True):
                if not st.session_state["srt_content"]:
                    st.error("Fix subtitle errors before burning.")
                elif target_language == "he" and not font_path:
                    st.error("Please provide a Hebrew font path for correct rendering.")
                else:
                    tmp_dir = st.session_state["temp_dir"]
                    srt_path = os.path.join(tmp_dir, "subtitles.srt")
                    output_path = os.path.join(tmp_dir, "output_burned.mp4")

                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state["srt_content"])

                    with st.spinner("Burning subtitles with FFmpeg..."):
                        try:
                            burn_subtitles(
                                input_video=st.session_state["uploaded_video_path"],
                                srt_path=srt_path,
                                output_path=output_path,
                                font_path=font_path or None,
                                is_rtl=(target_language == "he"),
                                font_size=font_size,
                            )

                            st.success("Done! Click below to download.")
                            with open(output_path, "rb") as vf:
                                video_bytes = vf.read()
                            st.download_button(
                                label="Download burned video",
                                data=video_bytes,
                                file_name="video_with_subtitles_burned.mp4",
                                mime="video/mp4",
                                use_container_width=True,
                            )
                        except (RuntimeError, FileNotFoundError) as e:
                            st.error(str(e))
