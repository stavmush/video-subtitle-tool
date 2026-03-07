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

from utils.autosave import clear_session, load_session, save_session
from utils.improve import improve_text_list
from utils.srt_utils import _str_to_timedelta, dataframe_to_srt, merge_srt_dataframes, parse_srt_to_dataframe, segments_to_dataframe
from utils.transcribe import transcribe_to_english, transcribe_video
from utils.translate import translate_segments, translate_text_list
from utils.video import burn_subtitles, embed_subtitles, embed_subtitles_multi

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
    "srt_mode": False,          # True when user loaded an existing SRT (skips transcribe/translate)
    "loaded_srt_name": None,    # Filename of last loaded SRT, to detect new uploads
    "loaded_video_name": None,  # Filename of last uploaded video, to detect new uploads
    "merge_srt_names": [],      # List of filenames from last merge, to detect new uploads
    "autosave_dismissed": False, # True once user has responded to the restore banner this session
}

for key, default in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _cleanup_temp(tmp_dir: str | None) -> None:
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _clear_editor_state() -> None:
    """Clear the data_editor's stored deltas before programmatic DataFrame updates.
    Without this, stale deltas from the old data get re-applied to the new DataFrame."""
    st.session_state.pop("subtitle_editor", None)


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
        clear_session()
        for key, default in STATE_DEFAULTS.items():
            st.session_state[key] = default
        st.rerun()

    st.caption("Models are downloaded on first use and cached locally.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Video Subtitle Tool")
st.caption("Transcribe, translate, edit, and export subtitles for your videos.")

# ── Autosave restore banner ───────────────────────────────────────────────────
_autosave = load_session()
if _autosave and not st.session_state["autosave_dismissed"] and not st.session_state["translation_done"]:
    saved_at = _autosave.get("saved_at", "unknown time")
    src = _autosave.get("source_filename") or "unknown file"
    st.warning(f"Unsaved session found — **{src}**, last saved {saved_at}")
    _rb_col, _db_col = st.columns(2)
    if _rb_col.button("Restore session", type="primary", use_container_width=True):
        _df = pd.DataFrame(_autosave["subtitles"])
        st.session_state["subtitles_df"] = _df
        st.session_state["srt_content"] = dataframe_to_srt(_df)
        st.session_state["srt_mode"] = True
        st.session_state["transcription_done"] = True
        st.session_state["translation_done"] = True
        st.session_state["autosave_dismissed"] = True
        _vp = _autosave.get("video_path", "")
        if _vp and os.path.isfile(_vp):
            if st.session_state["temp_dir"] is None:
                st.session_state["temp_dir"] = tempfile.mkdtemp(prefix="subtitle_tool_")
            st.session_state["uploaded_video_path"] = _vp
        _clear_editor_state()
        st.rerun()
    if _db_col.button("Discard", use_container_width=True):
        clear_session()
        st.session_state["autosave_dismissed"] = True
        st.rerun()
    st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Input (video or existing SRT)
# ═══════════════════════════════════════════════════════════════════════════════
st.header("1. Input")

tab_video, tab_srt, tab_embed, tab_merge = st.tabs(["Transcribe from video", "Load existing SRT", "Embed SRT into video", "Merge SRT files"])

# ── Tab A: upload video and transcribe ────────────────────────────────────────
with tab_video:
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mkv", "avi", "mov"],
        label_visibility="collapsed",
    )

    if uploaded_file and st.session_state.get("loaded_video_name") != uploaded_file.name:
        _cleanup_temp(st.session_state["temp_dir"])
        tmp_dir = tempfile.mkdtemp(prefix="subtitle_tool_")
        st.session_state["temp_dir"] = tmp_dir
        video_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["uploaded_video_path"] = video_path
        st.session_state["loaded_video_name"] = uploaded_file.name
        st.session_state["srt_mode"] = False
        st.session_state["transcription_done"] = False
        st.session_state["translation_done"] = False
        st.session_state["subtitles_df"] = None
        st.session_state["srt_content"] = None
        st.session_state["whisper_segments"] = None
        _clear_editor_state()
        st.rerun()

    if st.session_state["uploaded_video_path"] and not st.session_state["srt_mode"]:
        video_path = st.session_state["uploaded_video_path"]
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb <= 200:
            st.video(video_path)
        else:
            st.info(
                f"Video uploaded: **{os.path.basename(video_path)}** "
                f"({file_size_mb:.0f} MB) — preview skipped for large files to save RAM."
            )

# ── Tab B: load an existing SRT file ─────────────────────────────────────────
with tab_srt:
    uploaded_srt = st.file_uploader(
        "Choose an SRT file",
        type=["srt"],
        label_visibility="collapsed",
        key="srt_uploader",
    )

    if uploaded_srt and st.session_state.get("loaded_srt_name") != uploaded_srt.name:
        try:
            srt_text = uploaded_srt.read().decode("utf-8")
            df = parse_srt_to_dataframe(srt_text)
            if df.empty:
                st.error("The SRT file appears to be empty or could not be parsed.")
            else:
                if st.session_state["temp_dir"] is None:
                    st.session_state["temp_dir"] = tempfile.mkdtemp(prefix="subtitle_tool_")
                st.session_state["subtitles_df"] = df
                st.session_state["srt_content"] = srt_text
                st.session_state["translation_done"] = True
                st.session_state["transcription_done"] = True
                st.session_state["srt_mode"] = True
                st.session_state["loaded_srt_name"] = uploaded_srt.name
                save_session(df, uploaded_srt.name, st.session_state.get("uploaded_video_path"), target_language)
                _clear_editor_state()
                st.rerun()
        except Exception as e:
            st.error(f"Failed to parse SRT file: {e}")

    if st.session_state["srt_mode"]:
        st.success(f"SRT loaded — {len(st.session_state['subtitles_df'])} subtitles ready to edit.")

        st.markdown("**Want to embed/burn these subtitles into a video?** Upload it below:")
        uploaded_video_for_srt = st.file_uploader(
            "Choose a video file (optional)",
            type=["mp4", "mkv", "avi", "mov"],
            label_visibility="collapsed",
            key="video_for_srt",
        )
        if uploaded_video_for_srt and st.session_state["uploaded_video_path"] is None:
            video_path = os.path.join(st.session_state["temp_dir"], uploaded_video_for_srt.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video_for_srt.getbuffer())
            st.session_state["uploaded_video_path"] = video_path
            st.rerun()

        if st.session_state["uploaded_video_path"]:
            st.info(f"Video ready: **{os.path.basename(st.session_state['uploaded_video_path'])}**")

# ── Tab C: quick embed — SRT + video → output, no editor ─────────────────────
with tab_embed:
    st.caption("Upload an SRT file and a video, then embed the subtitles directly — no editing needed.")

    col_srt, col_vid = st.columns(2)

    with col_srt:
        st.markdown("**SRT file**")
        quick_srt = st.file_uploader(
            "SRT file",
            type=["srt"],
            label_visibility="collapsed",
            key="quick_srt",
        )

    with col_vid:
        st.markdown("**Video file**")
        quick_video = st.file_uploader(
            "Video file",
            type=["mp4", "mkv", "avi", "mov"],
            label_visibility="collapsed",
            key="quick_video",
        )

    if quick_srt and quick_video:
        ql_col_a, ql_col_b = st.columns(2)
        with ql_col_a:
            quick_primary_lang = st.selectbox(
                "Primary SRT language",
                ["en", "he"],
                format_func=lambda x: "English" if x == "en" else "Hebrew",
                key="quick_primary_lang",
            )
        with ql_col_b:
            quick_add_second = st.checkbox("Add a second subtitle track", key="quick_add_second")

        quick_srt_2 = None
        quick_second_lang = None
        if quick_add_second:
            qs2_col_a, qs2_col_b = st.columns(2)
            with qs2_col_a:
                quick_second_lang = st.selectbox(
                    "Second track language",
                    ["he", "en"],
                    format_func=lambda x: "Hebrew" if x == "he" else "English",
                    key="quick_second_lang",
                )
            with qs2_col_b:
                quick_srt_2 = st.file_uploader(
                    "Second SRT file",
                    type=["srt"],
                    label_visibility="collapsed",
                    key="quick_srt_2",
                )

        if st.button("Embed subtitles into video", type="primary", use_container_width=True):
            if quick_add_second and not quick_srt_2:
                st.error("Upload the second SRT file or uncheck 'Add a second subtitle track'.")
            else:
                quick_tmp = tempfile.mkdtemp(prefix="quick_embed_")
                try:
                    quick_srt_path = os.path.join(quick_tmp, "subtitles.srt")
                    quick_video_path = os.path.join(quick_tmp, quick_video.name)
                    quick_output_path = os.path.join(quick_tmp, "output_embedded.mp4")

                    with open(quick_srt_path, "wb") as f:
                        f.write(quick_srt.read())
                    with open(quick_video_path, "wb") as f:
                        f.write(quick_video.getbuffer())

                    with st.spinner("Embedding subtitles with FFmpeg..."):
                        if quick_add_second and quick_srt_2:
                            quick_srt_path_2 = os.path.join(quick_tmp, "subtitles_2.srt")
                            with open(quick_srt_path_2, "wb") as f:
                                f.write(quick_srt_2.read())
                            embed_subtitles_multi(
                                input_video=quick_video_path,
                                srt_tracks=[(quick_srt_path, quick_primary_lang), (quick_srt_path_2, quick_second_lang)],
                                output_path=quick_output_path,
                            )
                        else:
                            embed_subtitles(
                                input_video=quick_video_path,
                                srt_path=quick_srt_path,
                                output_path=quick_output_path,
                            )

                    st.success("Done!")
                    with open(quick_output_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.download_button(
                        label="Download video with embedded subtitles",
                        data=video_bytes,
                        file_name="video_with_subtitles.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(str(e))
                finally:
                    shutil.rmtree(quick_tmp, ignore_errors=True)
    elif quick_srt and not quick_video:
        st.info("Upload a video file to continue.")
    elif quick_video and not quick_srt:
        st.info("Upload an SRT file to continue.")

# ── Tab D: merge multiple SRT files into one ──────────────────────────────────
with tab_merge:
    st.caption("Upload two or more SRT files. Timestamps will be auto-offset so they chain sequentially.")

    merge_files = st.file_uploader(
        "Choose SRT files",
        type=["srt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="merge_srt_uploader",
    )

    if merge_files:
        st.markdown("**Merge order:**")
        for i, f in enumerate(merge_files, 1):
            st.markdown(f"{i}. {f.name}")

    if st.button(
        "Merge SRT files",
        type="primary",
        disabled=len(merge_files) < 2 if merge_files else True,
        use_container_width=True,
    ):
        try:
            dfs = []
            for f in merge_files:
                srt_text = f.read().decode("utf-8")
                df = parse_srt_to_dataframe(srt_text)
                if df.empty:
                    st.error(f"'{f.name}' could not be parsed or is empty.")
                    st.stop()
                dfs.append(df)

            merged_df = merge_srt_dataframes(dfs)

            if st.session_state["temp_dir"] is None:
                st.session_state["temp_dir"] = tempfile.mkdtemp(prefix="subtitle_tool_")
            st.session_state["subtitles_df"] = merged_df
            st.session_state["srt_content"] = dataframe_to_srt(merged_df)
            st.session_state["srt_mode"] = True
            st.session_state["transcription_done"] = True
            st.session_state["translation_done"] = True
            st.session_state["merge_srt_names"] = [f.name for f in merge_files]
            save_session(merged_df, ", ".join(f.name for f in merge_files), st.session_state.get("uploaded_video_path"), target_language)
            _clear_editor_state()
            st.rerun()
        except Exception as e:
            st.error(f"Merge failed: {e}")

    if st.session_state["srt_mode"] and st.session_state.get("merge_srt_names"):
        n = len(st.session_state["subtitles_df"])
        files = ", ".join(st.session_state["merge_srt_names"])
        st.success(f"Merged {len(st.session_state['merge_srt_names'])} files ({files}) — {n} subtitles ready to edit.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Transcribe  (skipped in SRT mode)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["uploaded_video_path"] and not st.session_state["srt_mode"]:
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
                save_session(st.session_state["subtitles_df"], st.session_state.get("loaded_video_name"), st.session_state.get("uploaded_video_path"), target_language)
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
# STEP 3 — Translate  (skipped in SRT mode)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["transcription_done"] and not st.session_state["srt_mode"]:
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
    _as_meta = load_session()
    if _as_meta:
        st.header("4. Edit Subtitles")
        st.caption(f"Last saved: {_as_meta['saved_at']}")
    else:
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

    # ── Stats bar ─────────────────────────────────────────────────────────────
    try:
        durations = [
            (_str_to_timedelta(str(row["end"])) - _str_to_timedelta(str(row["start"]))).total_seconds()
            for _, row in df.iterrows()
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        total_duration_str = str(df["end"].iloc[-1]) if not df.empty else "—"
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Subtitles", len(df))
        stat_col2.metric("Total duration", total_duration_str)
        stat_col3.metric("Avg on screen", f"{avg_duration:.1f}s")
    except Exception:
        pass  # don't let a bad timestamp break the editor

    # ── Find & replace ────────────────────────────────────────────────────────
    with st.expander("Find & replace"):
        fr_col_a, fr_col_b = st.columns(2)
        with fr_col_a:
            fr_search = st.text_input("Find", key="fr_search", placeholder="Search text...")
        with fr_col_b:
            fr_replace = st.text_input("Replace with", key="fr_replace", placeholder="Replacement...")
        fr_case = st.checkbox("Case sensitive", value=False, key="fr_case")
        if st.button("Replace all", use_container_width=True, disabled=not fr_search):
            before = df["text"].copy()
            new_text = df["text"].str.replace(fr_search, fr_replace, case=fr_case, regex=False)
            count = (new_text != before).sum()
            if count:
                new_df = df.copy()
                new_df["text"] = new_text
                st.session_state["subtitles_df"] = new_df
                _clear_editor_state()
                st.rerun()
            else:
                st.info(f'No matches found for "{fr_search}".')

    col_left, col_mid, col_right = st.columns([3, 2, 1])
    with col_left:
        st.caption("Fix grammar and fluency of English subtitles using a local AI model.")
        if st.button("Improve subtitles (grammar)", use_container_width=True):
            try:
                with st.spinner("Improving subtitles (downloading model on first run)..."):
                    improved = improve_text_list(df["text"].tolist())
                    new_df = df.copy()
                    new_df["text"] = improved
                    st.session_state["subtitles_df"] = new_df
                _clear_editor_state()
                st.rerun()
            except Exception as e:
                st.error(f"Improvement failed: {e}")
    with col_mid:
        tl_col_a, tl_col_b = st.columns(2)
        with tl_col_a:
            tl_src = st.selectbox(
                "From",
                ["en", "he"],
                format_func=lambda x: "English" if x == "en" else "Hebrew",
                key="tl_src",
                label_visibility="collapsed",
            )
        with tl_col_b:
            tl_tgt = st.selectbox(
                "To",
                ["he", "en"],
                format_func=lambda x: "Hebrew" if x == "he" else "English",
                key="tl_tgt",
                label_visibility="collapsed",
            )
        if st.button("Translate subtitles", use_container_width=True):
            if tl_src == tl_tgt:
                st.warning("Source and target language are the same.")
            else:
                try:
                    with st.spinner(f"Translating subtitles to {'Hebrew' if tl_tgt == 'he' else 'English'}..."):
                        texts = df["text"].tolist()
                        translated = translate_text_list(texts, tl_src, tl_tgt)
                        new_df = df.copy()
                        new_df["text"] = translated
                        st.session_state["subtitles_df"] = new_df
                    _clear_editor_state()
                    st.rerun()
                except Exception as e:
                    st.error(f"Translation failed: {e}")
    with col_right:
        if st.button("Renumber subtitles"):
            df = df.copy()
            df["index"] = range(1, len(df) + 1)
            st.session_state["subtitles_df"] = df
            _clear_editor_state()
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

    # Persist edits: if the user changed anything, bake the delta into session
    # state and clear it so it isn't re-applied on the next unrelated rerun
    # (stale deltas being re-applied is what causes the intermittent revert bug).
    try:
        dfs_equal = edited_df.equals(df)
    except Exception:
        dfs_equal = False
    if not dfs_equal:
        st.session_state["subtitles_df"] = edited_df
        save_session(
            edited_df,
            source_filename=st.session_state.get("loaded_srt_name") or st.session_state.get("loaded_video_name"),
            video_path=st.session_state.get("uploaded_video_path"),
            target_language=target_language,
        )
        _clear_editor_state()
        st.rerun()

    # Regenerate SRT from current state
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

        if not st.session_state["uploaded_video_path"]:
            st.info(
                "No video uploaded. Go to the **Load existing SRT** tab "
                "and upload a video there to enable embed/burn."
            )

        embed_tab, burn_tab = st.tabs(["Embed (Recommended)", "Burn (Hard subs)"])

        # ── Embed (soft track) ────────────────────────────────────────────────
        with embed_tab:
            st.caption(
                "Adds a subtitle track inside the MP4 container — **no re-encoding**. "
                "Selectable in VLC, IINA, QuickTime, and most players. Fast."
            )

            primary_lang = st.selectbox(
                "Primary track language",
                ["en", "he"],
                index=0 if target_language == "en" else 1,
                format_func=lambda x: "English" if x == "en" else "Hebrew",
                key="embed_primary_lang",
            )

            add_second_track = st.checkbox("Add a second subtitle track", key="embed_add_second")
            second_srt_content = None
            second_lang = None
            if add_second_track:
                second_lang = st.selectbox(
                    "Second track language",
                    ["he", "en"],
                    format_func=lambda x: "Hebrew" if x == "he" else "English",
                    key="embed_second_lang",
                )
                second_srt_file = st.file_uploader(
                    "Second SRT file",
                    type=["srt"],
                    label_visibility="collapsed",
                    key="embed_second_srt",
                )
                if second_srt_file:
                    second_srt_content = second_srt_file.read().decode("utf-8")

            if st.button("Embed subtitles", type="primary", use_container_width=True):
                if not st.session_state["uploaded_video_path"]:
                    st.error("Upload a video first (see the Load existing SRT tab).")
                elif not st.session_state["srt_content"]:
                    st.error("Fix subtitle errors before embedding.")
                elif add_second_track and not second_srt_content:
                    st.error("Upload the second SRT file or uncheck 'Add a second subtitle track'.")
                else:
                    tmp_dir = st.session_state["temp_dir"]
                    srt_path = os.path.join(tmp_dir, "subtitles.srt")
                    output_path = os.path.join(tmp_dir, "output_embedded.mp4")

                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state["srt_content"])

                    with st.spinner("Embedding subtitles with FFmpeg..."):
                        try:
                            if add_second_track and second_srt_content:
                                srt_path_2 = os.path.join(tmp_dir, "subtitles_2.srt")
                                with open(srt_path_2, "w", encoding="utf-8") as f:
                                    f.write(second_srt_content)
                                embed_subtitles_multi(
                                    input_video=st.session_state["uploaded_video_path"],
                                    srt_tracks=[(srt_path, primary_lang), (srt_path_2, second_lang)],
                                    output_path=output_path,
                                )
                            else:
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
                if not st.session_state["uploaded_video_path"]:
                    st.error("Upload a video first (see the Load existing SRT tab).")
                elif not st.session_state["srt_content"]:
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
