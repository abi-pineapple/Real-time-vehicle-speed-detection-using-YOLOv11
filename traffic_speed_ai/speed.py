import math
import time
from collections import deque, defaultdict


class SpeedEstimator:
    """
    Estimates real-world speed (km/h) of tracked vehicles.

    Calibration: user supplies a pixel distance that corresponds to a known
    real-world distance (meters). This gives meters-per-pixel, which is
    combined with elapsed time between frames to compute speed.
    """

    def __init__(self, pixel_distance=400, real_distance_m=20, fps=30, smoothing_window=5):
        self.set_calibration(pixel_distance, real_distance_m)
        self.fps = fps
        self.smoothing_window = smoothing_window
        # per-track history of (x, y, timestamp)
        self.history = defaultdict(lambda: deque(maxlen=30))
        # per-track smoothed speed readings
        self.speed_window = defaultdict(lambda: deque(maxlen=smoothing_window))

    def set_calibration(self, pixel_distance, real_distance_m):
        pixel_distance = max(1, pixel_distance)
        real_distance_m = max(0.1, real_distance_m)
        self.meters_per_pixel = real_distance_m / pixel_distance

    def set_fps(self, fps):
        if fps and fps > 0:
            self.fps = fps

    def update(self, track_id, center_x, center_y, frame_idx=None):
        """
        Feed a new observed center point for a track. Returns the current
        smoothed speed estimate in km/h (0 until enough history exists).
        """
        now = time.time()
        hist = self.history[track_id]
        hist.append((center_x, center_y, now, frame_idx))

        if len(hist) < 2:
            return 0.0

        (x1, y1, t1, f1), (x2, y2, t2, f2) = hist[-2], hist[-1]

        # Prefer frame-based elapsed time (more stable than wall clock for
        # recorded video processed faster/slower than real time).
        if f1 is not None and f2 is not None and self.fps > 0:
            elapsed = max(1, (f2 - f1)) / self.fps
        else:
            elapsed = max(t2 - t1, 1e-3)

        pixel_dist = math.hypot(x2 - x1, y2 - y1)
        meters = pixel_dist * self.meters_per_pixel
        speed_mps = meters / elapsed
        speed_kmh = speed_mps * 3.6

        # discard unrealistic spikes (tracking jitter / id switches)
        if speed_kmh > 250:
            speed_kmh = self.get_smoothed(track_id)

        window = self.speed_window[track_id]
        window.append(speed_kmh)
        return sum(window) / len(window)

    def get_smoothed(self, track_id):
        window = self.speed_window[track_id]
        return sum(window) / len(window) if window else 0.0

    def reset_track(self, track_id):
        self.history.pop(track_id, None)
        self.speed_window.pop(track_id, None)

    def reset_all(self):
        self.history.clear()
        self.speed_window.clear()
