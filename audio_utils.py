import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def probe_audio(path: Path) -> Dict[str, str]:
    """Inspect audio file metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})
    return {
        "duration": fmt.get("duration"),
        "sample_rate": stream.get("sample_rate"),
        "channels": stream.get("channels"),
        "format": fmt.get("format_long_name"),
    }


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: Optional[int] = None) -> None:
    """Transcode input audio to float32 WAV for processing."""
    args = ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-acodec", "pcm_f32le"]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    args += ["-map_metadata", "-1", str(output_path)]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.strip()}")


def export_audio(input_wav: Path, output_path: Path, fmt: str) -> None:
    """Export WAV to requested format."""
    fmt = fmt.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "wav":
        shutil.copyfile(input_wav, output_path)
        return
    if fmt == "mp3":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(output_path),
        ]
    elif fmt == "flac":
        cmd = ["ffmpeg", "-y", "-i", str(input_wav), "-vn", "-codec:a", "flac", str(output_path)]
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg export failed: {result.stderr.strip()}")


def hash_bytes_settings(file_bytes: bytes, settings: Dict[str, str]) -> str:
    """Hash file bytes and settings to generate a cache key."""
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def cleanup_old_runs(base_dir: Path, keep_last: int = 3) -> None:
    """Remove older run folders to reclaim disk."""
    if not base_dir.exists():
        return
    dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in dirs[keep_last:]:
        shutil.rmtree(stale, ignore_errors=True)


def suppress_residuals(
    instrumental_path: Path, vocals_path: Path, output_path: Path, strength: float = 0.2
) -> None:
    """
    Apply light residual suppression by subtracting a small portion of vocals
    from the instrumental stem.
    """
    inst_audio, sr = sf.read(instrumental_path, dtype="float32", always_2d=True)
    voc_audio, _ = sf.read(vocals_path, dtype="float32", always_2d=True)
    min_len = min(len(inst_audio), len(voc_audio))
    inst_audio = inst_audio[:min_len]
    voc_audio = voc_audio[:min_len]
    cleaned = inst_audio - strength * voc_audio
    cleaned = np.clip(cleaned, -1.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, cleaned, sr, subtype="PCM_16")


def make_temp_run_dir(base_dir: Path) -> Path:
    """Create a temporary directory under base_dir."""
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=base_dir))
