import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict
import zipfile

import streamlit as st

import audio_utils
import separator

st.set_page_config(page_title="EveryInstrument - Vocal Remover", layout="wide")

BASE_RUN_DIR = Path(tempfile.gettempdir()) / "everyinstrument_runs"
audio_utils.cleanup_old_runs(BASE_RUN_DIR, keep_last=4)

if "results" not in st.session_state:
    st.session_state["results"] = None


def _write_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    return Path(tmp.name)


@st.cache_data(show_spinner=False)
def process_audio(file_bytes: bytes, filename: str, options_json: str) -> Dict[str, object]:
    options = json.loads(options_json)
    cache_key = audio_utils.hash_bytes_settings(file_bytes, options)
    run_dir = BASE_RUN_DIR / cache_key
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / filename
    with open(input_path, "wb") as f:
        f.write(file_bytes)

    working_wav = run_dir / "input.wav"
    audio_utils.convert_to_wav(input_path, working_wav)

    separation_dir = run_dir / "separation"
    sep_result = separator.separate_audio(
        working_wav,
        separation_dir,
        stems_mode=options["stems_mode"],
        quality=options["quality"],
        use_gpu=options["use_gpu"],
        residual_suppression=options["residual_suppression"],
    )

    exports_dir = run_dir / "exports"
    exports_dir.mkdir(exist_ok=True)
    exports = {}
    for name, stem_path in sep_result["stems"].items():
        target = exports_dir / f"{name}.{options['output_format']}"
        audio_utils.export_audio(stem_path, target, options["output_format"])
        exports[name] = target

    zip_path = None
    if len(exports) > 1:
        zip_path = exports_dir / "stems.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, path in exports.items():
                zf.write(path, arcname=path.name)

    return {
        "cache_key": cache_key,
        "input_path": input_path,
        "working_wav": working_wav,
        "engine": sep_result["engine"],
        "stems": sep_result["stems"],
        "exports": exports,
        "zip": zip_path,
        "log": sep_result["log"],
        "options": options,
    }


def audio_bytes(path: Path) -> bytes:
    return path.read_bytes()


def render_results(results: Dict[str, object]) -> None:
    st.subheader("Results")
    st.caption(f"Engine: {results['engine']}, Quality: {results['options']['quality'].title()}")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Original**")
        original_preview = results.get("working_wav", results["input_path"])
        st.audio(audio_bytes(original_preview), format="audio/wav")
    with cols[1]:
        st.markdown("**Instrumental**")
        inst = results["exports"].get("instrumental") or results["exports"].get("accompaniment")
        if inst:
            st.audio(audio_bytes(inst))
        else:
            st.info("No instrumental available")
    with cols[2]:
        st.markdown("**Vocals**")
        vocals = results["exports"].get("vocals")
        if vocals:
            st.audio(audio_bytes(vocals))
        else:
            st.info("Vocals not generated")

    st.markdown("---")
    st.markdown("**Downloads**")
    dl_cols = st.columns(3)
    with dl_cols[0]:
        inst_path = results["exports"].get("instrumental")
        st.download_button(
            "Download Instrumental",
            data=audio_bytes(inst_path) if inst_path else None,
            file_name=f"instrumental.{results['options']['output_format']}",
            disabled=inst_path is None,
        )
    with dl_cols[1]:
        voc_path = results["exports"].get("vocals")
        st.download_button(
            "Download Vocals",
            data=audio_bytes(voc_path) if voc_path else None,
            file_name=f"vocals.{results['options']['output_format']}",
            disabled=voc_path is None,
        )
    with dl_cols[2]:
        if results["zip"]:
            st.download_button("Download All Stems (zip)", results["zip"].read_bytes(), file_name="stems.zip")
        else:
            st.info("Zip available when multiple stems are generated.")

    st.markdown("**Logs**")
    st.code(results["log"] or "No logs")


def main() -> None:
    st.title("EveryInstrument: High-Quality Vocal Remover")
    st.write("Upload a song, pick quality, and download instrumentals or stems.")

    ffmpeg_ok = audio_utils.check_ffmpeg()
    if not ffmpeg_ok:
        st.error(
            "ffmpeg is required but not detected. Install ffmpeg and ensure it is on PATH "
            "(Windows: add to Environment Variables, Ubuntu: `sudo apt-get install ffmpeg`)."
        )

    with st.expander("File upload", expanded=True):
        uploaded = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "flac"])
        st.caption("Guidance: keep uploads under ~15 minutes or 150 MB to avoid long processing times.")
        file_info = {}
        if uploaded:
            tmp_path = _write_upload_to_temp(uploaded)
            try:
                file_info = audio_utils.probe_audio(tmp_path)
            except Exception as exc:
                st.warning(f"Could not read audio metadata: {exc}")
            finally:
                os.unlink(tmp_path)
            st.write(
                f"Size: {uploaded.size / 1e6:.2f} MB, "
                f"Duration: {file_info.get('duration', 'unknown')} s, "
                f"Sample rate: {file_info.get('sample_rate', 'unknown')} Hz, "
                f"Channels: {file_info.get('channels', 'unknown')}"
            )

    st.markdown("---")
    st.subheader("Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        quality = st.selectbox("Quality / Speed", ["Balanced", "Best", "Fast"], index=0)
        output_format = st.selectbox("Output format", ["wav", "mp3", "flac"], index=0)
    with col2:
        stems_label = st.radio(
            "Stems",
            [
                "Instrumental only (vocals removed)",
                "Two stems: vocals + accompaniment",
                "Four stems: vocals, drums, bass, other",
            ],
            index=0,
        )
        stems_mode = "instrumental"
        if stems_label.startswith("Two"):
            stems_mode = "two_stems"
        elif stems_label.startswith("Four"):
            stems_mode = "four_stems"
    with col3:
        gpu_available = separator.is_cuda_available()
        use_gpu = st.checkbox("Use GPU if available", value=gpu_available, disabled=not gpu_available)
        residual = st.checkbox("Light vocal residual suppression", value=False)

    st.caption("Tip: WAV/FLAC output keeps highest fidelity. Large songs may take time; prefer Best quality for final renders.")

    run_button = st.button("Run separation", type="primary", disabled=not uploaded or not ffmpeg_ok)
    reset_button = st.button("Reset", type="secondary")

    if reset_button:
        st.session_state["results"] = None
        st.experimental_rerun()

    if run_button and uploaded:
        progress = st.progress(5)
        st.write("Preparing audio...")
        options = {
            "quality": quality.lower(),
            "output_format": output_format.lower(),
            "stems_mode": stems_mode,
            "use_gpu": bool(use_gpu),
            "residual_suppression": bool(residual),
        }
        try:
            progress.progress(25)
            result = process_audio(uploaded.getbuffer().tobytes(), uploaded.name, json.dumps(options))
            progress.progress(100)
            st.session_state["results"] = result
            st.success("Separation finished.")
        except Exception as exc:
            st.error(f"Separation failed: {exc}")

    if st.session_state["results"]:
        render_results(st.session_state["results"])


if __name__ == "__main__":
    main()
