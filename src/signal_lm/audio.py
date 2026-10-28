"""Windows WASAPI loopback, bounded in-memory capture with a heartbeat lease."""
from collections import deque
from datetime import datetime, timedelta, timezone
import platform
import threading
import time
import uuid

import numpy as np

from .analysis import analyze
from .signal import Signal


def backend():
    if platform.system() != "Windows":
        raise RuntimeError("La capture audio système nécessite Windows / WASAPI.")
    try:
        import pyaudiowpatch
    except ImportError as exc:
        raise RuntimeError('Installez la dépendance audio : python -m pip install -e ".[audio]"') from exc
    return pyaudiowpatch


def devices():
    module = backend()
    with module.PyAudio() as pa:
        try:
            default = int(pa.get_default_wasapi_loopback()["index"])
        except (OSError, LookupError):
            default = None
        return [{"id": int(d["index"]), "name": d["name"],
                 "sample_rate": int(d["defaultSampleRate"]),
                 "channels": int(d["maxInputChannels"]), "default": int(d["index"]) == default}
                for d in pa.get_loopback_device_info_generator()]


def analyze_audio(samples, rate, zone, device, end_utc, sequence, discontinuities):
    channels = tuple(f"audio_{i + 1}" for i in range(samples.shape[1]))
    signal = Signal(samples, rate, channels, ("FS",) * len(channels))
    result = analyze(signal, include_spectrum=True)
    indices = np.unique(np.linspace(0, len(samples) - 1, min(1800, len(samples))).astype(int))
    result["waveform"] = {"time_s": (indices / rate).tolist(),
                          "values": samples[indices].T.tolist(), "display_samples": len(indices)}
    result["capture"] = {"zone": zone, "source": "wasapi_loopback", "device": device,
                         "window_start_utc": (end_utc - timedelta(seconds=len(samples) / rate)).isoformat(),
                         "window_end_utc": end_utc.isoformat(), "sequence": sequence,
                         "discontinuities": discontinuities,
                         "timestamp_basis": "host_callback_receipt_estimate",
                         "persisted": False, "clipped": bool(np.any(np.abs(samples) >= 1))}
    return result


class AudioCapture:
    def __init__(self):
        self.control = threading.RLock()
        self.buffer_lock = threading.Lock()
        self.blocks = deque()
        self.frames = 0
        self.sequence = 0
        self.discontinuities = 0
        self.last_data = 0
        self.end_utc = None
        self.session = None
        self.pa = self.stream = None
        self.closed = threading.Event()
        self.monitor = None

    def start(self, device_id, zone):
        if not isinstance(zone, str) or not zone.strip() or len(zone) > 80:
            raise ValueError("Une zone de 1 à 80 caractères est requise.")
        with self.control:
            if self.session:
                raise RuntimeError("Une capture est déjà active. Arrêtez-la dans son onglet.")
            module = backend()
            self.pa = module.PyAudio()
            try:
                info = self.pa.get_device_info_by_index(int(device_id))
                if not info.get("isLoopbackDevice"):
                    raise ValueError("Sélectionnez une sortie WASAPI loopback.")
                self.rate = int(info["defaultSampleRate"])
                self.channels = int(info["maxInputChannels"])
                if not 4 <= self.rate <= 192000 or not 1 <= self.channels <= 8:
                    raise ValueError("Format non pris en charge : maximum 192 kHz et 8 canaux.")
                self.zone, self.device = zone.strip(), info["name"]
                with self.buffer_lock:
                    self.blocks.clear()
                    self.frames = self.sequence = self.discontinuities = 0
                    self.last_data = 0
                    self.end_utc = None

                def callback(raw, frame_count, time_info, status):
                    values = np.frombuffer(raw, dtype=np.float32).reshape(-1, self.channels).copy()
                    with self.buffer_lock:
                        if status:
                            self.blocks.clear()
                            self.frames = 0
                            self.discontinuities += 1
                        self.blocks.append(values)
                        self.frames += len(values)
                        while self.blocks and self.frames - len(self.blocks[0]) >= self.rate:
                            self.frames -= len(self.blocks.popleft())
                        self.sequence += 1
                        self.end_utc = datetime.now(timezone.utc)
                        self.last_data = time.monotonic()
                    return (None, module.paContinue)

                self.stream = self.pa.open(format=module.paFloat32, channels=self.channels,
                    rate=self.rate, input=True, input_device_index=int(device_id),
                    frames_per_buffer=1024, stream_callback=callback, start=False)
                self.session = uuid.uuid4().hex
                self.heartbeat = time.monotonic()
                self.stream.start_stream()
                if self.monitor is None:
                    self.monitor = threading.Thread(target=self._watch, daemon=True)
                    self.monitor.start()
                return {"session": self.session, "sample_rate": self.rate,
                        "channels": self.channels, "zone": self.zone, "device": self.device}
            except Exception:
                self._stop()
                raise

    def frame(self, session):
        with self.control:
            if not self.session or session != self.session:
                raise ValueError("Capture arrêtée ou session expirée. Redémarrez la capture.")
            self.heartbeat = time.monotonic()
            with self.buffer_lock:
                if self.frames < self.rate or time.monotonic() - self.last_data > 3:
                    return {"waiting": True, "message": "En attente d’une seconde d’audio. Lancez un son sur la sortie choisie."}
                samples = np.concatenate(self.blocks)[-self.rate:].copy()
                end, sequence, gaps = self.end_utc, self.sequence, self.discontinuities
            return analyze_audio(samples, self.rate, self.zone, self.device, end, sequence, gaps)

    def stop(self, session):
        with self.control:
            if self.session and session != self.session:
                raise ValueError("Cette capture appartient à un autre onglet.")
            self._stop()
        return {"stopped": True}

    def _stop(self):
        self.session = None
        try:
            if self.stream:
                self.stream.close()
        finally:
            self.stream = None
            if self.pa:
                self.pa.terminate()
                self.pa = None
            with self.buffer_lock:
                self.blocks.clear()
                self.frames = 0

    def _watch(self):
        while not self.closed.wait(1):
            self.expire_idle()

    def expire_idle(self):
        with self.control:
            if self.session and time.monotonic() - self.heartbeat > 15:
                self._stop()

    def close(self):
        self.closed.set()
        with self.control:
            self._stop()


capture = AudioCapture()
