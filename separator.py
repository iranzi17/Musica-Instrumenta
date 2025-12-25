import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import audio_utils


def is_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:
        return False


def _base_command() -> str:
    return shutil.which("python") or "python"


def _select_model(quality: str) -> str:
    quality = quality.lower()
    if quality == "fast":
        return "htdemucs_6s"
    if quality == "best":
        return "mdx_extra_q"
    return "htdemucs"


def _run_process(cmd) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Process failed")
    return (result.stdout or "") + (result.stderr or "")


def run_demucs(
    input_wav: Path,
    output_dir: Path,
    stems_mode: str,
    quality: str = "balanced",
    use_gpu: bool = True,
) -> Dict[str, object]:
    model = _select_model(quality)
    cmd = [
        _base_command(),
        "-m",
        "demucs.separate",
        "-n",
        model,
        "-o",
        str(output_dir),
    ]
    if stems_mode in {"instrumental", "two_stems"}:
        cmd += ["--two-stems", "vocals"]
    device = "cuda" if use_gpu and is_cuda_available() else "cpu"
    cmd += ["-d", device, str(input_wav)]
    log = _run_process(cmd)
    stem_dir = _find_stem_dir(output_dir, input_wav.stem)
    stems = _map_demucs_outputs(stem_dir, stems_mode)
    return {"stems": stems, "log": log}


def _find_stem_dir(output_dir: Path, track_name: str) -> Path:
    candidates = [p for p in output_dir.rglob(track_name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("Demucs output not found")
    return candidates[0]


def _map_demucs_outputs(stem_dir: Path, stems_mode: str) -> Dict[str, Path]:
    stems: Dict[str, Path] = {}
    if stems_mode in {"instrumental", "two_stems"}:
        vocals = stem_dir / "vocals.wav"
        no_vocals = stem_dir / "no_vocals.wav"
        if not no_vocals.exists():
            alt = stem_dir / "accompaniment.wav"
            if alt.exists():
                no_vocals = alt
        if not vocals.exists() or not no_vocals.exists():
            raise FileNotFoundError("Expected two-stem outputs missing")
        stems["vocals"] = vocals
        stems["instrumental"] = no_vocals
    else:
        for name in ["vocals", "drums", "bass", "other"]:
            path = stem_dir / f"{name}.wav"
            if not path.exists():
                raise FileNotFoundError(f"Missing stem {name}")
            stems[name] = path
    return stems


def run_spleeter(input_wav: Path, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["spleeter", "separate", "-o", str(output_dir), "-p", "spleeter:2stems", str(input_wav)]
    log = _run_process(cmd)
    stem_dir = output_dir / input_wav.stem
    vocals = stem_dir / "vocals.wav"
    acc = stem_dir / "accompaniment.wav"
    if not vocals.exists() or not acc.exists():
        raise FileNotFoundError("Spleeter output missing")
    return {"stems": {"vocals": vocals, "instrumental": acc}, "log": log}


def separate_audio(
    input_wav: Path,
    work_dir: Path,
    stems_mode: str,
    quality: str = "balanced",
    use_gpu: bool = True,
    residual_suppression: bool = False,
) -> Dict[str, object]:
    log_lines = []
    work_dir.mkdir(parents=True, exist_ok=True)
    demucs_dir = work_dir / "demucs"
    spleeter_dir = work_dir / "spleeter"
    engine = "demucs"
    try:
        log_lines.append(f"Running Demucs ({quality}) on {input_wav.name}")
        demucs_out = run_demucs(input_wav, demucs_dir, stems_mode, quality, use_gpu)
        log_lines.append(demucs_out["log"])
        stems = demucs_out["stems"]
    except Exception as demucs_err:
        log_lines.append(f"Demucs failed: {demucs_err}")
        if stems_mode == "four_stems":
            raise
        log_lines.append("Falling back to Spleeter 2-stems")
        engine = "spleeter"
        spleeter_out = run_spleeter(input_wav, spleeter_dir)
        log_lines.append(spleeter_out["log"])
        stems = spleeter_out["stems"]
    if residual_suppression and "instrumental" in stems and "vocals" in stems:
        cleaned_path = work_dir / "post" / "instrumental_clean.wav"
        audio_utils.suppress_residuals(stems["instrumental"], stems["vocals"], cleaned_path)
        stems["instrumental"] = cleaned_path
        log_lines.append("Applied light residual suppression")
    return {
        "engine": engine,
        "stems": stems,
        "log": "\n".join(log_lines),
    }
