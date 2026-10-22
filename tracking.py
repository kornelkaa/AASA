from typing import List, Dict, Any
import numpy as np


class TemporalTracker:
    """
    Multi-frame Temporal Tracking module.
    Maintains a rolling window history of object distances across multiple time steps
    to smooth out sensor noise and calculate accurate relative velocities.
    """
    def __init__(self, history_size: int = 4):
        self.history_size = history_size
        # Dictionary storing up to `history_size` past distance values per object
        self.distance_history: Dict[str, List[float]] = {}

    def track_and_estimate_velocity(
        self,
        current_detections: List[Dict[str, Any]],
        time_delta_sec: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Calculates relative velocity using multi-frame sliding window differentiation.
        """
        updated_detections = []

        for det in current_detections:
            label = det['label']
            curr_dist = det.get('distance_m')

            if curr_dist is not None:
                # Initialize history list for new objects
                if label not in self.distance_history:
                    self.distance_history[label] = []

                # Append current distance reading
                self.distance_history[label].append(curr_dist)

                # Restrict memory buffer to max history size
                if len(self.distance_history[label]) > self.history_size:
                    self.distance_history[label].pop(0)

                history = self.distance_history[label]

                # Velocity estimation requires at least 2 historical frames
                if len(history) >= 2:
                    # Calculate average distance change per frame step
                    dist_deltas = [history[i] - history[i + 1] for i in range(len(history) - 1)]
                    avg_dist_delta = float(np.mean(dist_deltas))  # Smooth out sensor noise

                    speed_ms = avg_dist_delta / time_delta_sec
                    speed_kmh = round(speed_ms * 3.6, 1)

                    if speed_kmh > 1.5:
                        motion_state = "APPROACHING"
                    elif speed_kmh < -1.5:
                        motion_state = "RECEDING"
                    else:
                        motion_state = "STATIONARY"

                    det['speed_kmh'] = abs(speed_kmh)
                    det['motion_state'] = motion_state
                else:
                    det['speed_kmh'] = 0.0
                    det['motion_state'] = "INITIALIZING"
            else:
                det['speed_kmh'] = 0.0
                det['motion_state'] = "UNKNOWN"

            updated_detections.append(det)

        return updated_detections


if __name__ == '__main__':
    tracker = TemporalTracker(history_size=3)

    # Simulating 3 consecutive frame detections for a truck
    simulated_frames = [
        [{'label': 'truck', 'distance_m': 15.0}],  # Frame 0
        [{'label': 'truck', 'distance_m': 13.6}],  # Frame 1 (0.5s later)
        [{'label': 'truck', 'distance_m': 12.3}]   # Frame 2 (1.0s later)
    ]

    print("--- Multi-Frame Tracking Simulation ---")
    for frame_idx, detections in enumerate(simulated_frames):
        results = tracker.track_and_estimate_velocity(detections, time_delta_sec=0.5)
        obj = results[0]
        print(f"Frame {frame_idx}: Distance={obj['distance_m']}m | State={obj['motion_state']} | Speed={obj['speed_kmh']} km/h")