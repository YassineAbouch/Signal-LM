"""Loopback workspace: physical analysis, read-only inference and explicit annotations."""
import argparse
import csv
import io
import json
import platform
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .analysis import analyze
from .signal import Signal
from .audio import capture, devices as audio_devices
from .audio_file import analyze_file, MAX_FILE_BYTES
from . import learning, model_test

STATIC = Path(__file__).with_name("static")
MAX_BODY = 2 * 1024 * 1024


def analyze_request(data):
    rate = float(data.get("sample_rate", 4096))
    components = None
    component_values = []
    if data.get("source", "demo") == "csv":
        columns = data.get("columns", [])
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Sélectionnez des colonnes uniques.")
        reader = csv.DictReader(io.StringIO(data["csv"].lstrip("\ufeff")),
                                delimiter=data.get("delimiter", ","))
        if not reader.fieldnames or any(c not in reader.fieldnames for c in columns):
            raise ValueError("Colonnes absentes de l’en-tête CSV.")
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                rows.append([float(row[c]) for c in columns])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Valeur numérique invalide, ligne {line}.") from exc
            if len(rows) > 100000:
                raise ValueError("Limite : 100 000 échantillons.")
        signal = Signal(rows, rate, tuple(columns), tuple(data.get("units", [])))
    else:
        if not np.isfinite(rate) or not 4 <= rate <= 100000:
            raise ValueError("La fréquence d’échantillonnage de démo doit être entre 4 et 100 000 Hz.")
        raw_components = data.get("components", [{"frequency": data.get("frequency", 120),
                                                  "amplitude": data.get("amplitude", 1)}])
        if not isinstance(raw_components, list) or not 1 <= len(raw_components) <= 16:
            raise ValueError("Le signal doit contenir entre 1 et 16 composantes.")
        components = []
        t = np.arange(int(rate)) / rate
        for index, component in enumerate(raw_components, 1):
            if not isinstance(component, dict):
                raise ValueError(f"Composante {index} : objet attendu.")
            frequency = float(component.get("frequency", 120))
            amplitude = float(component.get("amplitude", 1))
            phase = float(component.get("phase_deg", 0))
            if not np.isfinite(frequency) or not 0 < frequency < rate / 2:
                raise ValueError(f"Composante {index} : fréquence positive et inférieure à fs/2 requise.")
            if not np.isfinite(amplitude) or not 0 < amplitude <= 1000:
                raise ValueError(f"Composante {index} : amplitude attendue dans ]0, 1000].")
            if not np.isfinite(phase) or not -360 <= phase <= 360:
                raise ValueError(f"Composante {index} : phase attendue entre -360 et 360 degrés.")
            components.append({"frequency": frequency, "amplitude": amplitude, "phase_deg": phase})
            component_values.append(amplitude * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase)))
        signal = Signal(np.sum(component_values, axis=0), rate,
                        ("synthetic" if len(components) == 1 else "composite",), ("a.u.",))
    if signal.samples.shape[1] > 8:
        raise ValueError("Limite : 8 canaux.")
    result = analyze(signal, include_spectrum=True)
    # Display reductions only; measurements always use the full signal.
    indices = np.unique(np.linspace(0, len(signal.samples) - 1,
                                   min(1800, len(signal.samples))).astype(int))
    result["waveform"] = {"time_s": (indices / rate).tolist(),
                          "values": signal.samples[indices].T.tolist(),
                          "display_samples": len(indices)}
    if components is not None:
        result["synthetic"] = {"method": "sum_of_sinusoids", "components": components}
        result["waveform"]["components"] = [values[indices].tolist() for values in component_values]
    return result


def doctor():
    import torch
    from .model import ModelConfig, build_virgin
    model = build_virgin(ModelConfig())
    config = model.config
    architecture = []
    channels = config.input_channels
    for index, width in enumerate(config.widths, 1):
        architecture.append({"name": f"Conv1d {index}", "input_channels": channels,
                             "output_channels": width, "kernel_size": config.kernel_size,
                             "activation": "GELU"})
        channels = width
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "status": "untrained", "training_steps": 0,
            "config": asdict(config), "architecture": architecture,
            "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "receptive_field_samples": 1 + len(config.widths) * (config.kernel_size - 1),
            "platform": platform.platform(),
            "model_source": "fresh_default_initialization",
            "gpu_memory_bytes": (torch.cuda.get_device_properties(0).total_memory
                                 if torch.cuda.is_available() else None)}


class Handler(BaseHTTPRequestHandler):
    def local_host(self):
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in allowed:
            self.respond(403, {"error": "Hôte local requis."})
            return False
        return True

    def respond(self, status, content, mime="application/json; charset=utf-8"):
        raw = content if isinstance(content, bytes) else json.dumps(content, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not self.local_host():
            return
        if self.path == "/api/doctor":
            self.respond(200, doctor())
            return
        if self.path in {"/api/learning/stats", "/api/model-test/info"}:
            try:
                self.respond(200, learning.stats() if self.path.endswith("stats") else model_test.info())
            except (ValueError, RuntimeError, OSError) as exc:
                self.respond(503, {"error": str(exc)})
            return
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/learn": ("learn.html", "text/html; charset=utf-8"),
                 "/learn.js": ("learn.js", "text/javascript; charset=utf-8"),
                 "/test-model": ("test-model.html", "text/html; charset=utf-8"),
                 "/test-model.js": ("test-model.js", "text/javascript; charset=utf-8"),
                 "/audio": ("audio.html", "text/html; charset=utf-8"),
                 "/audio.js": ("audio.js", "text/javascript; charset=utf-8"),
                 "/charts.js": ("charts.js", "text/javascript; charset=utf-8"),
                 "/model": ("model.html", "text/html; charset=utf-8"),
                 "/model/": ("model.html", "text/html; charset=utf-8"),
                 "/model.js": ("model.js", "text/javascript; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path not in files:
            self.respond(404, {"error": "Ressource introuvable."})
            return
        name, mime = files[self.path]
        self.respond(200, (STATIC / name).read_bytes(), mime)

    def do_POST(self):
        if not self.local_host():
            return
        binary_paths = {"/api/audio/file", "/api/learning/save", "/api/model-test/file"}
        if self.path not in {"/api/analyze", "/api/audio/devices", "/api/audio/start", "/api/audio/frame", "/api/audio/stop", "/api/model-test/example"} | binary_paths:
            self.respond(404, {"error": "Ressource introuvable."})
            return
        # Reject cross-origin requests to this local service.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.respond(403, {"error": "Origine refusée."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            limit = MAX_FILE_BYTES if self.path in binary_paths else MAX_BODY
            if not 0 < size <= limit:
                self.respond(413, {"error": f"Requête limitée à {limit // 1024 // 1024} Mio."})
                return
            if self.path == "/api/learning/save":
                self.respond(200, learning.save_annotation(self.rfile.read(size),
                    unquote(self.headers.get("X-Audio-Name", "audio")),
                    zone=unquote(self.headers.get("X-Audio-Zone", "zone_01")),
                    window_start=self.headers.get("X-Audio-Start", "0"),
                    selection_start=self.headers.get("X-Selection-Start", "0"),
                    selection_end=self.headers.get("X-Selection-End", "1"),
                    label=unquote(self.headers.get("X-Learning-Label", "")),
                    meaning=unquote(self.headers.get("X-Learning-Meaning", "")),
                    machine_id=unquote(self.headers.get("X-Learning-Machine", ""))))
                return
            if self.path == "/api/model-test/file":
                analysis, signal = analyze_file(self.rfile.read(size),
                    unquote(self.headers.get("X-Audio-Name", "audio")),
                    unquote(self.headers.get("X-Audio-Zone", "zone_01")),
                    self.headers.get("X-Audio-Start", "0"), return_signal=True)
                self.respond(200, {"analysis": analysis, "model": model_test.predict(signal)})
                return
            if self.path == "/api/audio/file":
                self.respond(200, analyze_file(self.rfile.read(size),
                    unquote(self.headers.get("X-Audio-Name", "audio")),
                    unquote(self.headers.get("X-Audio-Zone", "zone_01")),
                    self.headers.get("X-Audio-Start", "0")))
                return
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Objet JSON attendu.")
            if self.path == "/api/model-test/example":
                result = model_test.test_example(data["record_id"])
            elif self.path == "/api/audio/devices":
                result = {"devices": audio_devices()}
            elif self.path == "/api/audio/start":
                result = capture.start(data["device_id"], data["zone"])
            elif self.path == "/api/audio/frame":
                result = capture.frame(data["session"])
            elif self.path == "/api/audio/stop":
                result = capture.stop(data.get("session"))
            else:
                result = analyze_request(data)
            self.respond(200, result)
        except (ValueError, TypeError, KeyError, OverflowError, csv.Error) as exc:
            self.respond(400, {"error": str(exc)})
        except (RuntimeError, OSError, LookupError) as exc:
            self.respond(503, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Signal-LM local web workspace")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Signal-LM : http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()
        server.server_close()


if __name__ == "__main__":
    main()
