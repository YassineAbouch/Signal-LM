import numpy as np

from signal_lm.data import generate_dataset
from signal_lm.data import quality
from signal_lm.data.windows import window_record


def test_regeneration_checks_parent_slice_for_windows(tmp_path):
    records = generate_dataset({"dataset_id": "quality-test", "specialist": "vibration",
        "sample_rate_hz": 64, "n_samples": 64, "channel_names": ["x"], "units": ["a.u."],
        "seed": 1, "n_parents": 1, "snr_db_levels": [None],
        "recipes": [{"recipe_id": "sine", "task_type": "event_classification", "labels": ["sine"],
                     "components": [{"type": "sine", "frequency_hz": 8, "amplitude": 1}]}]}, tmp_path)
    records[0].split = "train"
    children = window_record(tmp_path, records[0], window_samples=16, hop_samples=16)
    problems, _ = quality.run_quality(tmp_path, records + children, verify_regeneration=True)
    assert not [p for p in problems if p["severity"] == "error"]


def test_one_float32_ulp_is_explicit_warning_larger_difference_is_error(monkeypatch):
    from signal_lm.data.schema import Record
    record = Record("r", "s", "vibration", "event_classification", "r.npz", 64, 4,
                    ["x"], ["a.u."], labels=["sine"], group_id="g", is_synthetic=True,
                    generation={"seed": 1, "components": []})
    arrays = {key: np.ones((4, 1), dtype=np.float64) for key in ["clean", "noise", "mix"]}
    for steps, expected in [(1, "warning"), (2, "error")]:
        fresh = {key: value.copy() for key, value in arrays.items()}
        for _ in range(steps):
            fresh["noise"][0, 0] = np.nextafter(np.float32(fresh["noise"][0, 0]), np.float32(2))
        monkeypatch.setattr(quality, "regenerate_record", lambda record: fresh)
        problems = quality._check_synthetic(record, arrays, True)
        assert len(problems) == 1 and problems[0]["severity"] == expected
