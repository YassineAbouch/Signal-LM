"""Reproducible synthetic signal generators for pipeline verification.

Every generator is a pure function of (parameters, seed). A synthetic example
is built as ``mix = clean + noise`` where ``clean`` is the sum of the
deterministic components and ``noise`` is the noise component scaled to a
target SNR. Clean, noise and mix are all kept, together with the exact
parameters and seed, so any file can be regenerated bit-for-bit.

Labels produced here are *descriptive* ("periodic_impulses", "chirp"), never a
mechanical diagnosis: a sine with impulses is not a "confirmed bearing fault".

SNR definition (documented in docs/DATA_SCHEMA.md):
    SNR_dB = 10 * log10( mean(clean**2) / mean(noise**2) )
computed over the full record and all channels. For short events an
"active" SNR is also reported, using only the samples inside annotated
events. Both measured values are stored in the manifest and checked by
``quality``.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

NYQUIST_MARGIN = 0.9   # components must satisfy f <= 0.9 * fs / 2 unless aliasing is requested

# --------------------------------------------------------------------------- RNG


def derive_seed(*parts: Any) -> int:
    """Stable 63-bit seed derived from arbitrary JSON-serialisable parts."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") & ((1 << 63) - 1)


def rng_from(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))


# --------------------------------------------------------------------------- helpers


def _check_frequency(f_hz: float, fs: float, allow_aliasing: bool) -> None:
    if f_hz < 0:
        raise ValueError("Frequencies must be non-negative.")
    if not allow_aliasing and f_hz > NYQUIST_MARGIN * fs / 2:
        raise ValueError(f"Frequency {f_hz} Hz exceeds {NYQUIST_MARGIN} * Nyquist for fs={fs} Hz; "
                         "set allow_aliasing=True to simulate aliasing explicitly.")


def _time(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / fs


def power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def snr_db(clean: np.ndarray, noise: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """10*log10(P_clean / P_noise); None when either power is zero."""
    if mask is not None:
        clean, noise = clean[mask], noise[mask]
    p_s, p_n = power(clean), power(noise)
    if p_s <= 0 or p_n <= 0:
        return None
    return 10.0 * math.log10(p_s / p_n)


def scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, target_db: float,
                       mask: np.ndarray | None = None) -> np.ndarray:
    """Scale ``noise`` so that snr_db(clean, noise, mask) == target_db."""
    ref_clean = clean if mask is None else clean[mask]
    ref_noise = noise if mask is None else noise[mask]
    p_s, p_n = power(ref_clean), power(ref_noise)
    if p_s <= 0:
        raise ValueError("Cannot define SNR for a zero-power clean signal.")
    if p_n <= 0:
        raise ValueError("Cannot scale a zero-power noise.")
    return noise * math.sqrt(p_s / (p_n * 10 ** (target_db / 10)))


# --------------------------------------------------------------------------- components
# Each component returns (samples[n], events) where events is a list of
# (label, start_sample, end_sample) for time-localised content.

Component = tuple[np.ndarray, list[tuple[str, int, int]]]


def sine(n: int, fs: float, *, frequency_hz: float, amplitude: float = 1.0,
         phase_rad: float = 0.0, allow_aliasing: bool = False, **_) -> Component:
    _check_frequency(frequency_hz, fs, allow_aliasing)
    return amplitude * np.sin(2 * np.pi * frequency_hz * _time(n, fs) + phase_rad), []


def multi_sine(n: int, fs: float, *, frequencies_hz: list[float], amplitudes: list[float],
               phases_rad: list[float] | None = None, allow_aliasing: bool = False, **_) -> Component:
    phases = phases_rad or [0.0] * len(frequencies_hz)
    if not len(frequencies_hz) == len(amplitudes) == len(phases):
        raise ValueError("frequencies_hz, amplitudes and phases_rad must have equal length.")
    x = np.zeros(n)
    for f, a, p in zip(frequencies_hz, amplitudes, phases):
        x += sine(n, fs, frequency_hz=f, amplitude=a, phase_rad=p, allow_aliasing=allow_aliasing)[0]
    return x, []


def harmonics(n: int, fs: float, *, fundamental_hz: float, n_harmonics: int = 5,
              amplitude: float = 1.0, decay: float = 0.6, phase_rad: float = 0.0,
              allow_aliasing: bool = False, **_) -> Component:
    """Fundamental plus harmonics k*f0 with amplitude*decay**(k-1).

    Harmonics above the Nyquist margin are dropped (not aliased) and the
    number actually kept is returned through ``describe_harmonics``.
    """
    x = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        f = k * fundamental_hz
        if f > NYQUIST_MARGIN * fs / 2 and not allow_aliasing:
            break
        x += sine(n, fs, frequency_hz=f, amplitude=amplitude * decay ** (k - 1),
                  phase_rad=phase_rad * k, allow_aliasing=allow_aliasing)[0]
    return x, []


def harmonics_kept(fs: float, fundamental_hz: float, n_harmonics: int, allow_aliasing: bool = False) -> int:
    kept = 0
    for k in range(1, n_harmonics + 1):
        if k * fundamental_hz > NYQUIST_MARGIN * fs / 2 and not allow_aliasing:
            break
        kept += 1
    return kept


def chirp(n: int, fs: float, *, f_start_hz: float, f_end_hz: float, amplitude: float = 1.0,
          method: str = "linear", phase_rad: float = 0.0, allow_aliasing: bool = False, **_) -> Component:
    _check_frequency(f_start_hz, fs, allow_aliasing)
    _check_frequency(f_end_hz, fs, allow_aliasing)
    t = _time(n, fs)
    duration = n / fs
    if method == "linear":
        phase = 2 * np.pi * (f_start_hz * t + 0.5 * (f_end_hz - f_start_hz) / duration * t ** 2)
    elif method == "exponential":
        if f_start_hz <= 0 or f_end_hz <= 0:
            raise ValueError("Exponential chirp needs positive frequencies.")
        ratio = f_end_hz / f_start_hz
        phase = 2 * np.pi * f_start_hz * duration / np.log(ratio) * (ratio ** (t / duration) - 1)
    else:
        raise ValueError("method must be 'linear' or 'exponential'.")
    return amplitude * np.sin(phase + phase_rad), []


def amplitude_modulation(n: int, fs: float, *, carrier_hz: float, modulation_hz: float,
                         amplitude: float = 1.0, depth: float = 0.5, phase_rad: float = 0.0,
                         allow_aliasing: bool = False, **_) -> Component:
    _check_frequency(carrier_hz, fs, allow_aliasing)
    _check_frequency(modulation_hz, fs, allow_aliasing)
    if not 0 <= depth <= 1:
        raise ValueError("depth must be within [0, 1].")
    t = _time(n, fs)
    envelope = 1 + depth * np.sin(2 * np.pi * modulation_hz * t)
    return amplitude * envelope * np.sin(2 * np.pi * carrier_hz * t + phase_rad), []


def _impulse_shape(width: int, kind: str) -> np.ndarray:
    if width < 1:
        raise ValueError("Impulse width must be >= 1 sample.")
    if kind == "rect":
        return np.ones(width)
    if kind == "triangle":  # interior points only, so width 1 gives a unit peak
        return 1 - np.abs(np.linspace(-1, 1, width + 2)[1:-1])
    if kind == "half_sine":
        return np.sin(np.pi * (np.arange(width) + 0.5) / width)
    raise ValueError("kind must be rect, triangle or half_sine.")


def impulses(n: int, fs: float, *, positions_samples: list[int], amplitude: float = 1.0,
             width_samples: int = 1, kind: str = "rect", label: str = "impulse",
             polarity: list[int] | None = None, **_) -> Component:
    """Impulses at explicit sample positions; events cover [pos, pos+width)."""
    x = np.zeros(n)
    events = []
    shape = _impulse_shape(width_samples, kind)
    signs = polarity or [1] * len(positions_samples)
    for pos, sign in zip(positions_samples, signs):
        if not 0 <= pos < n:
            raise ValueError(f"Impulse position {pos} outside [0, {n}).")
        end = min(n, pos + width_samples)
        x[pos:end] += sign * amplitude * shape[: end - pos]
        events.append((label, int(pos), int(end)))
    return x, events


def periodic_impulse_positions(n: int, fs: float, rate_hz: float, offset_samples: int = 0,
                               jitter_samples: int = 0, rng: np.random.Generator | None = None) -> list[int]:
    period = fs / rate_hz
    if period < 1:
        raise ValueError("Impulse rate exceeds the sample rate.")
    positions = []
    k = 0
    while True:
        pos = int(round(offset_samples + k * period))
        if jitter_samples and rng is not None:
            pos += int(rng.integers(-jitter_samples, jitter_samples + 1))
        if pos >= n:
            break
        if pos >= 0:
            positions.append(pos)
        k += 1
    return positions


def damped_oscillation(n: int, fs: float, *, frequency_hz: float, onset_sample: int,
                       amplitude: float = 1.0, decay_rate_per_s: float = 50.0,
                       label: str = "damped_burst", phase_rad: float = 0.0,
                       allow_aliasing: bool = False, **_) -> Component:
    """Exponentially decaying sine starting at onset; event ends at 1 % of amplitude."""
    _check_frequency(frequency_hz, fs, allow_aliasing)
    if not 0 <= onset_sample < n:
        raise ValueError("onset_sample outside the record.")
    if decay_rate_per_s <= 0:
        raise ValueError("decay_rate_per_s must be > 0.")
    x = np.zeros(n)
    t = _time(n - onset_sample, fs)
    x[onset_sample:] = amplitude * np.exp(-decay_rate_per_s * t) * np.sin(2 * np.pi * frequency_hz * t + phase_rad)
    end = onset_sample + int(math.ceil(math.log(100) / decay_rate_per_s * fs))
    return x, [(label, int(onset_sample), int(min(n, end)))]


def white_noise(n: int, fs: float, *, rng: np.random.Generator, std: float = 1.0, **_) -> Component:
    return rng.normal(0.0, std, size=n), []


def colored_noise(n: int, fs: float, *, rng: np.random.Generator, exponent: float = 1.0,
                  std: float = 1.0, **_) -> Component:
    """Noise with PSD ~ 1/f**exponent (0 white, 1 pink, 2 brown; negative = blue).

    Built by shaping the spectrum of white noise, then rescaled to ``std``.
    """
    white = rng.normal(0.0, 1.0, size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    gain = np.ones_like(freqs)
    gain[1:] = freqs[1:] ** (-exponent / 2)
    gain[0] = 0.0  # no DC
    x = np.fft.irfft(spectrum * gain, n=n)
    x = x - x.mean()
    scale = np.std(x)
    return (x / scale * std) if scale > 0 else x, []


def band_noise(n: int, fs: float, *, rng: np.random.Generator, low_hz: float, high_hz: float,
               std: float = 1.0, **_) -> Component:
    """Band-limited white noise obtained by spectral masking."""
    if not 0 <= low_hz < high_hz <= fs / 2:
        raise ValueError("Band must satisfy 0 <= low < high <= fs/2.")
    white = rng.normal(0.0, 1.0, size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    x = np.fft.irfft(spectrum, n=n)
    scale = np.std(x)
    return (x / scale * std) if scale > 0 else x, []


def drift(n: int, fs: float, *, slope_per_s: float, offset: float = 0.0, **_) -> Component:
    return offset + slope_per_s * _time(n, fs), []


def level_step(n: int, fs: float, *, step_sample: int, step_amplitude: float,
               label: str = "level_change", **_) -> Component:
    if not 0 < step_sample < n:
        raise ValueError("step_sample must be inside (0, n).")
    x = np.zeros(n)
    x[step_sample:] = step_amplitude
    return x, [(label, int(step_sample), int(n))]


def saturate(x: np.ndarray, limit: float) -> tuple[np.ndarray, float]:
    """Hard clipping at +/-limit; returns (clipped, fraction_of_clipped_samples)."""
    if limit <= 0:
        raise ValueError("limit must be > 0.")
    clipped = np.clip(x, -limit, limit)
    fraction = float(np.mean(np.abs(x) >= limit))
    return clipped, fraction


def noise_burst(n: int, fs: float, *, rng: np.random.Generator, onset_sample: int,
                duration_samples: int, low_hz: float, high_hz: float, amplitude: float = 1.0,
                modulation_hz: float = 0.0, label: str = "noise_burst", **_) -> Component:
    """Band-limited noise burst with raised-cosine edges and optional AM.

    Descriptive label only (e.g. "noise_burst_modulated"); it is not a
    confirmed friction or rubbing diagnosis.
    """
    if not 0 <= onset_sample < n or duration_samples < 4:
        raise ValueError("Burst must start inside the record and last >= 4 samples.")
    end = min(n, onset_sample + duration_samples)
    length = end - onset_sample
    burst, _ = band_noise(length, fs, rng=rng, low_hz=low_hz, high_hz=high_hz, std=1.0)
    ramp = max(1, length // 10)
    envelope = np.ones(length)
    envelope[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
    envelope[-ramp:] = envelope[:ramp][::-1]
    if modulation_hz > 0:
        envelope *= 0.5 + 0.5 * np.sin(2 * np.pi * modulation_hz * _time(length, fs))
    x = np.zeros(n)
    x[onset_sample:end] = amplitude * burst * envelope
    return x, [(label, int(onset_sample), int(end))]


COMPONENTS = {
    "sine": sine, "multi_sine": multi_sine, "harmonics": harmonics, "chirp": chirp,
    "amplitude_modulation": amplitude_modulation, "impulses": impulses,
    "damped_oscillation": damped_oscillation, "drift": drift, "level_step": level_step,
    "noise_burst": noise_burst,
}
NEEDS_RNG = {"noise_burst"}
NOISES = {"white_noise": white_noise, "colored_noise": colored_noise, "band_noise": band_noise}


# --------------------------------------------------------------------------- example

def render(spec: dict[str, Any]) -> dict[str, Any]:
    """Render one fully-resolved example spec into arrays and measured quantities.

    ``spec`` keys: n_samples, sample_rate_hz, components (list of dicts with
    ``type`` + parameters, all concrete numbers), noise (dict or None with
    ``type``, parameters, ``snr_db`` or ``std``), seed, optional
    ``saturation_limit``. Returned arrays have shape [samples, 1] (float64).
    """
    n = int(spec["n_samples"])
    fs = float(spec["sample_rate_hz"])
    seed = int(spec["seed"])
    rng = np.random.default_rng(seed)
    clean = np.zeros(n)
    events: list[tuple[str, int, int]] = []
    for index, comp in enumerate(spec.get("components", [])):
        params = {k: v for k, v in comp.items() if k != "type"}
        if comp["type"] in NEEDS_RNG:
            params["rng"] = np.random.default_rng(derive_seed(seed, "component", index))
        x, ev = COMPONENTS[comp["type"]](n, fs, **params)
        clean += x
        events.extend(ev)

    noise = np.zeros(n)
    noise_spec = spec.get("noise")
    snr_target = None
    snr_mode = None
    if noise_spec:
        params = {k: v for k, v in noise_spec.items() if k not in ("type", "snr_db", "snr_reference")}
        noise, _ = NOISES[noise_spec["type"]](n, fs, rng=rng, **params)
        snr_target = noise_spec.get("snr_db")
        snr_mode = noise_spec.get("snr_reference", "global")
        if snr_target is not None:
            mask = _active_mask(n, events) if snr_mode == "active" else None
            if mask is not None and not mask.any():
                mask = None
            noise = scale_noise_to_snr(clean, noise, float(snr_target), mask)

    mix = clean + noise
    saturation_fraction = 0.0
    limit = spec.get("saturation_limit")
    if limit is not None:
        mix, saturation_fraction = saturate(mix, float(limit))

    active = _active_mask(n, events)
    measured = {
        "snr_db_measured": snr_db(clean, noise),
        "snr_db_measured_active": snr_db(clean, noise, active) if active.any() else None,
        "snr_reference": snr_mode,
        "clean_rms": float(np.sqrt(power(clean))),
        "noise_rms": float(np.sqrt(power(noise))),
        "mix_peak_abs": float(np.max(np.abs(mix))),
        "saturation_fraction": saturation_fraction,
    }
    return {"clean": clean[:, None], "noise": noise[:, None], "mix": mix[:, None],
            "events": events, "snr_db_target": snr_target, "measured": measured}


def _active_mask(n: int, events: list[tuple[str, int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for _, start, end in events:
        mask[start:end] = True
    return mask


# --------------------------------------------------------------------------- recipes

def _sample(rng: np.random.Generator, value: Any) -> Any:
    """Resolve a parameter: scalar -> itself; [lo, hi] -> uniform; {"choice": [...]} -> pick."""
    if isinstance(value, dict) and "choice" in value:
        options = value["choice"]
        return options[int(rng.integers(len(options)))]
    if isinstance(value, dict) and "log_uniform" in value:
        lo, hi = value["log_uniform"]
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    if isinstance(value, dict) and "int_uniform" in value:
        lo, hi = value["int_uniform"]
        return int(rng.integers(lo, hi + 1))
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        return float(rng.uniform(value[0], value[1]))
    return value


def resolve_component(rng: np.random.Generator, template: dict[str, Any], n: int, fs: float) -> dict[str, Any]:
    """Turn a recipe component template (with ranges) into concrete numbers."""
    comp = {k: _sample(rng, v) for k, v in template.items()}
    ctype = comp["type"]
    if ctype == "impulses" and "positions_samples" not in comp:
        if "rate_hz" in comp:
            comp["positions_samples"] = periodic_impulse_positions(
                n, fs, comp.pop("rate_hz"), offset_samples=int(rng.integers(0, max(1, int(fs / 50)))),
                jitter_samples=int(comp.pop("jitter_samples", 0)), rng=rng)
        else:
            count = int(comp.pop("count", 1))
            comp["positions_samples"] = sorted(int(p) for p in rng.integers(0, n, size=count))
        comp["width_samples"] = int(comp.get("width_samples", 1))
    if ctype == "damped_oscillation" and "onset_sample" not in comp:
        # *_fraction values were already drawn by _sample (ranges -> float)
        comp["onset_sample"] = int(float(comp.pop("onset_fraction", 0.3)) * n)
    if ctype == "noise_burst":
        if "onset_sample" not in comp:
            comp["onset_sample"] = int(float(comp.pop("onset_fraction", 0.3)) * n)
        if "duration_samples" not in comp:
            comp["duration_samples"] = int(float(comp.pop("duration_fraction", 0.2)) * n)
        comp["duration_samples"] = int(comp["duration_samples"])
    if ctype == "level_step" and "step_sample" not in comp:
        comp["step_sample"] = int(float(comp.pop("step_fraction", 0.5)) * n)
    if ctype == "harmonics":
        comp["n_harmonics"] = int(comp["n_harmonics"])
    return comp


def build_example_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a dataset config into fully-resolved, regenerable example specs.

    For each recipe, ``n_parents`` parent realisations are drawn (each with its
    own component parameters). Every parent is rendered once per entry of
    ``snr_db_levels`` (``None`` = clean, no noise); all variants of one parent
    share ``group_id`` so they are never split apart.
    """
    n = int(config["n_samples"])
    fs = float(config["sample_rate_hz"])
    base_seed = int(config["seed"])
    specs = []
    for recipe in config["recipes"]:
        rid = recipe["recipe_id"]
        for parent_index in range(int(recipe.get("n_parents", config.get("n_parents", 8)))):
            rng = rng_from(base_seed, rid, parent_index, "params")
            components = [resolve_component(rng, c, n, fs) for c in recipe["components"]]
            noise_template = recipe.get("noise", config.get("noise"))
            noise_params = {k: _sample(rng, v) for k, v in (noise_template or {}).items()
                            if k not in ("snr_db",)}
            saturation = recipe.get("saturation_limit")
            if saturation is not None:
                saturation = _sample(rng, saturation)
            levels = recipe.get("snr_db_levels", config.get("snr_db_levels", [None]))
            group_id = f"{config['dataset_id']}/{rid}/p{parent_index:04d}"
            for level in levels:
                noise = None
                if noise_template is not None and level is not None:
                    noise = dict(noise_params, snr_db=float(level))
                level_tag = "clean" if level is None else f"snr{level:+.0f}dB".replace("+", "p").replace("-", "m")
                spec = {
                    "record_id": f"{group_id}/{level_tag}".replace("/", "__"),
                    "group_id": group_id,
                    "recipe_id": rid,
                    "parent_index": parent_index,
                    "n_samples": n, "sample_rate_hz": fs,
                    "components": components,
                    "noise": noise,
                    "saturation_limit": saturation,
                    "seed": derive_seed(base_seed, rid, parent_index, "noise"),
                    "labels": list(recipe["labels"]),
                    "task_type": recipe["task_type"],
                    "specialist": config["specialist"],
                    "channel_names": list(config.get("channel_names", ["synthetic"])),
                    "units": list(config.get("units", ["a.u."])),
                }
                specs.append(spec)
    return specs
