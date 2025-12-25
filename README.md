# EveryInstrument – High-Quality Vocal Remover

Streamlit app for fast, reliable vocal removal and stem extraction using Demucs (primary) with a Spleeter 2-stem fallback. Runs locally on Windows or Ubuntu with GPU or CPU.

## Features
- Upload MP3/WAV/M4A/FLAC and auto-convert to float32 WAV internally.
- Quality/Speed presets → Demucs models (`htdemucs_6s`, `htdemucs`, `mdx_extra_q`).
- Stems: Instrumental only, 2 stems (vocals + accompaniment), or 4 stems (vocals/drums/bass/other).
- Optional light residual suppression to reduce leftover vocals.
- Output formats: WAV (lossless), MP3 320 kbps, FLAC. Zip provided when multiple stems exist.
- GPU toggle with CUDA detection; disk-based caching to avoid recomputation.
- Preview players plus download buttons for stems and logs in the UI.

## Requirements
- Python 3.10+ (recommended)
- `ffmpeg` on PATH (ffmpeg + ffprobe)
- For GPU: CUDA-capable GPU with correct NVIDIA drivers/CUDA toolkit + matching `torch` wheel

## Install
```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Ubuntu
pip install -r requirements.txt
```
> Note: The base install targets Python 3.11 (see `runtime.txt`). Demucs is the primary engine. Spleeter fallback is **not** installed by default for Streamlit Cloud compatibility.

### ffmpeg install
- **Windows**: Install from https://www.gyan.dev/ffmpeg/builds/ (full build). Add `bin` folder to PATH. Reopen terminal.
- **Ubuntu**: `sudo apt-get update && sudo apt-get install -y ffmpeg`

## Run
```bash
streamlit run app.py
```
Open the provided local URL in your browser.

## Deploy on Streamlit Community Cloud
1. Commit/push the repo (keep `requirements.txt` and `packages.txt`).
2. In Streamlit Cloud, create a new app pointing to this repo and `app.py`.
3. The platform will install `ffmpeg` and `libsndfile1` from `packages.txt` and all Python deps from `requirements.txt`.
4. GPU is not available on Streamlit Cloud, so run in CPU mode (the GPU toggle will be off).
5. Spleeter fallback is not installed on Streamlit Cloud. Demucs remains the primary engine; optionally install Spleeter locally (Python 3.8/3.9) if you need that fallback.

## Usage
1. Upload an audio file (shows duration/sample rate/channels).
2. Choose Quality/Speed, stems, output format, GPU toggle, and optional residual suppression.
3. Click **Run separation**. Progress + logs display. Downloads and previews appear after completion.
4. Use **Reset** to clear results between songs.

## Model mapping
- **Fast** → `htdemucs_6s`
- **Balanced (default)** → `htdemucs`
- **Best** → `mdx_extra_q`
Fallback: Spleeter 2-stem only if Demucs fails or weights are unavailable.

## Tips for best quality
- Prefer WAV/FLAC output when mastering; MP3 320 kbps for sharing.
- Use **Best** quality for final renders; **Fast** for drafts.
- Enable GPU if available; CPU works but is slower.
- Clean inputs (no clipping) yield better separation.

## Troubleshooting
- **ffmpeg missing**: Ensure `ffmpeg` and `ffprobe` run from the terminal.
- **First Demucs run is slow**: Model weights download once per model.
- **GPU not used**: Confirm `torch.cuda.is_available()` in Python; update NVIDIA drivers.
- **Spleeter errors**: It is only used as fallback; ensure TensorFlow installs correctly for your platform.
- **Long tracks**: Separation can take several minutes; keep the browser tab open.

## Limitations
- Very long or high-sample-rate tracks increase runtime and disk usage.
- Spleeter fallback is lower quality than Demucs and limited to 2 stems; it is not installed by default and unavailable on Streamlit Cloud.
- This app runs locally; ensure sufficient RAM and disk space in the temp directory.
