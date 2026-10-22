from experimental.hazard_model import HazardService
from experimental.tracking import TemporalTracker


def test_hazard_score_is_zero_without_distance():
    service = HazardService()
    detection = {"label": "car", "box": [0, 0, 100, 100], "distance_m": None}

    assert service.compute_hazard(detection) == 0.0


def test_hazard_score_is_between_zero_and_one():
    service = HazardService()
    detection = {"label": "person", "box": [100, 200, 300, 500], "distance_m": 5.4}

    score = service.compute_hazard(detection)

    assert 0.0 <= score <= 1.0


def test_tracker_reports_approaching_truck():
    tracker = TemporalTracker(history_size=3)
    frames = [15.0, 13.6, 12.3]

    results = None
    for distance in frames:
        results = tracker.track_and_estimate_velocity(
            [{"label": "truck", "distance_m": distance}], time_delta_sec=0.5
        )

    assert results[0]["motion_state"] == "APPROACHING"
    assert results[0]["speed_kmh"] > 0
