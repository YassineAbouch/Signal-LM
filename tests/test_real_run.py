import numpy as np
import pytest

from signal_lm.training.real_run import aggregate_clips, binary_metrics, thresholds_from_validation


def test_binary_metrics_include_false_alarms_and_tie_aware_auc():
    report = binary_metrics([0, 0, 1, 1], [.1, .8, .7, .9], .75)
    assert (report["tp"], report["fp"], report["fn"], report["tn"]) == (1, 1, 1, 1)
    assert report["f1"] == report["false_positive_rate"] == .5
    assert report["roc_auc"] == .75
    assert binary_metrics([0, 1], [.2, .2], .2)["roc_auc"] == .5
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0, np.nan], .5)


def test_clip_threshold_uses_reference_validation_and_mean_windows():
    windows = []
    for parent, target, values in [("a", 0, [1, 3]), ("b", 0, [5, 7]), ("c", 1, [1000, 2000])]:
        for score in values:
            windows.append({"parent_record_id": parent, "machine_id": "validation_machine",
                "target": target, "neural_score": score, "rms_score": score / 10})
    clips = aggregate_clips(windows)
    assert [r["neural_score"] for r in clips] == [2, 6, 1500]
    thresholds = thresholds_from_validation(windows, clips)
    assert thresholds["window_neural_score"] == pytest.approx(6.7)
    assert thresholds["clip_neural_score"] == pytest.approx(5.8)
    assert thresholds["clip_rms_score"] == pytest.approx(.58)
    with pytest.raises(ValueError):
        thresholds_from_validation(windows[-2:], clips[-1:])


def test_clip_aggregation_rejects_conflicting_labels():
    rows = [{"parent_record_id": "a", "machine_id": "m", "target": label,
             "neural_score": 1., "rms_score": .1} for label in [0, 1]]
    with pytest.raises(ValueError):
        aggregate_clips(rows)
