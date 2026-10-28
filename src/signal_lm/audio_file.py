"""Decode a bounded window from uploaded media through memory-only pipes."""
from datetime import datetime, timezone
import json
import shutil
import subprocess
import threading

import numpy as np

from .audio import analyze_audio

MAX_FILE_BYTES = 50 * 1024 * 1024
decode_lock = threading.Lock()


def run_media(command, raw):
    try:
        result = subprocess.run(command, input=raw, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Décodage trop long. Essayez un fichier plus court.") from exc
    if result.returncode:
        raise ValueError("Fichier illisible, format non pris en charge ou piste audio absente.")
    return result.stdout


def analyze_file(raw, filename, zone, start=0):
    if not 0 < len(raw) <= MAX_FILE_BYTES:
        raise ValueError("Le fichier doit contenir au maximum 50 Mio.")
    if not isinstance(zone, str) or not zone.strip() or len(zone) > 80:
        raise ValueError("Une zone de 1 à 80 caractères est requise.")
    start = float(start)
    if not np.isfinite(start) or not 0 <= start <= 86400:
        raise ValueError("Position attendue entre 0 et 86 400 secondes.")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg et ffprobe doivent être installés et accessibles dans PATH.")
    if not decode_lock.acquire(blocking=False):
        raise RuntimeError("Un fichier est déjà en cours d’analyse. Réessayez ensuite.")
    try:
        metadata = json.loads(run_media([ffprobe, "-v", "error", "-protocol_whitelist", "pipe",
            "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
            "-of", "json", "-i", "pipe:0"], raw))
        streams = metadata.get("streams", [])
        if not streams:
            raise ValueError("Ce fichier ne contient aucune piste audio.")
        rate = int(streams[0]["sample_rate"])
        channels = int(streams[0]["channels"])
        if not 4 <= rate <= 192000 or not 1 <= channels <= 8:
            raise ValueError("Format limité à 192 kHz et 8 canaux, sans rééchantillonnage.")
        decoded = run_media([ffmpeg, "-v", "error", "-nostdin", "-protocol_whitelist", "pipe",
            "-i", "pipe:0", "-ss", str(start), "-t", "1", "-map", "0:a:0", "-vn",
            "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"], raw)
        samples = np.frombuffer(decoded, dtype="<f4").reshape(-1, channels)
        if len(samples) < 4:
            raise ValueError("Pas assez d’échantillons à cette position. Choisissez une position plus tôt.")
        if len(samples) > rate:
            samples = samples[:rate]
        now = datetime.now(timezone.utc)
        result = analyze_audio(samples, rate, zone.strip(), "uploaded_file", now, 0, 0)
        result.pop("capture")
        result["audio_file"] = {"name": str(filename)[:240], "zone": zone.strip(),
            "window_start_s": start, "window_end_s": start + len(samples) / rate,
            "analyzed_at_utc": now.isoformat(), "acquisition_time_utc": None,
            "source": "uploaded_file_first_audio_stream", "persisted": False,
            "clipped": bool(np.any(np.abs(samples) >= 1))}
        return result
    finally:
        decode_lock.release()
