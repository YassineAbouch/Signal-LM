"""Loopback-only web workspace. No training or uploaded-file persistence."""
import argparse
import csv
import io
import json
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .analysis import analyze
from .signal import Signal

STATIC = Path(__file__).with_name("static")
MAX_BODY = 2 * 1024 * 1024


def analyze_request(data):
    rate = float(data.get("sample_rate", 4096))
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
        frequency = float(data.get("frequency", 120))
        amplitude = float(data.get("amplitude", 1))
        if not np.isfinite(rate) or not 4 <= rate <= 100000:
            raise ValueError("La fréquence d’échantillonnage de démo doit être entre 4 et 100 000 Hz.")
        if not np.isfinite(frequency) or not 0 < frequency < rate / 2:
            raise ValueError("La fréquence du signal doit être positive et inférieure à fs/2.")
        if not np.isfinite(amplitude) or not 0 < amplitude <= 1000:
            raise ValueError("Amplitude attendue : ]0, 1000].")
        t = np.arange(int(rate)) / rate
        signal = Signal(amplitude * np.sin(2 * np.pi * frequency * t), rate,
                        ("synthetic",), ("a.u.",))
    if signal.samples.shape[1] > 8:
        raise ValueError("Limite : 8 canaux.")
    result = analyze(signal, include_spectrum=True)
    # Display reductions only; measurements always use the full signal.
    indices = np.unique(np.linspace(0, len(signal.samples) - 1,
                                   min(1800, len(signal.samples))).astype(int))
    result["waveform"] = {"time_s": (indices / rate).tolist(),
                          "values": signal.samples[indices].T.tolist(),
                          "display_samples": len(indices)}
    return result


def doctor():
    import torch
    from .model import ModelConfig, build_virgin
    model = build_virgin(ModelConfig())
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "status": "untrained", "training_steps": 0}


class Handler(BaseHTTPRequestHandler):
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
        if self.path == "/api/doctor":
            self.respond(200, doctor())
            return
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path not in files:
            self.respond(404, {"error": "Ressource introuvable."})
            return
        name, mime = files[self.path]
        self.respond(200, (STATIC / name).read_bytes(), mime)

    def do_POST(self):
        if self.path != "/api/analyze":
            self.respond(404, {"error": "Ressource introuvable."})
            return
        # Reject cross-origin requests to this local service.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.respond(403, {"error": "Origine refusée."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                self.respond(413, {"error": "Requête limitée à 2 Mio."})
                return
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Objet JSON attendu.")
            self.respond(200, analyze_request(data))
        except (ValueError, TypeError, KeyError, OverflowError, csv.Error) as exc:
            self.respond(400, {"error": str(exc)})


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
        server.server_close()


if __name__ == "__main__":
    main()
