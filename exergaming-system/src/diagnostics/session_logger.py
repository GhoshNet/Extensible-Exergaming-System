"""
Session Logger — captures per-frame diagnostics and writes JSON + CSV summaries.

One JSON file is written per session to sessions/<timestamp>_<exercise>.json.
A rolling CSV summary row is appended to sessions/sessions_summary.csv.

All file I/O happens in end_session() so frame logging never blocks the
processing thread. list.append() is GIL-safe for concurrent calls.
"""
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionLogger:
    """Records diagnostic data for one exercise session."""

    def __init__(self, log_dir: str = "sessions") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._session_id: Optional[str] = None
        self._exercise_name: str = ""
        self._camera_resolution: str = "unknown"
        self._started_at: Optional[datetime] = None
        self._start_perf: float = 0.0
        self._frames: List[Dict[str, Any]] = []
        self._active: bool = False
        self._skipped_frames: int = 0   # frames where side_used == "none"
        self._warned_frames: int = 0    # frames that had at least one quality warning

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(self, exercise_name: str, camera_resolution: str = "unknown") -> None:
        """Begin recording. Resets any previous in-memory data."""
        self._started_at = datetime.now()
        self._session_id = self._started_at.strftime("%Y%m%d_%H%M%S")
        self._exercise_name = exercise_name
        self._camera_resolution = camera_resolution
        self._start_perf = time.perf_counter()
        self._frames = []
        self._active = True
        self._skipped_frames = 0
        self._warned_frames = 0
        print(f"[Logger] Session started — {exercise_name}")

    def log_frame(self, frame_data: Dict[str, Any]) -> None:
        """Append one frame's diagnostics. Safe to call from any thread."""
        if not self._active:
            return
        record = dict(frame_data)
        record["timestamp_ms"] = round((time.perf_counter() - self._start_perf) * 1000, 1)
        self._frames.append(record)
        if record.get("side_used") == "none":
            self._skipped_frames += 1
        if record.get("quality_warnings"):
            self._warned_frames += 1

    def end_session(self) -> Optional[str]:
        """
        Stop recording, compute summary stats, and write files.

        Returns the JSON file path on success, None if nothing was recorded.
        """
        if not self._active:
            return None
        self._active = False

        if not self._frames:
            print("[Logger] Session ended with no frames — nothing written.")
            return None

        ended_at = datetime.now()
        duration_s = time.perf_counter() - self._start_perf

        metadata = self._build_metadata(ended_at, duration_s)

        try:
            json_path = self._write_json(metadata)
            self._append_csv_summary(metadata)
            print(
                f"[Logger] Saved → {json_path.name}  "
                f"({metadata['total_reps']} reps, "
                f"{metadata['total_frames']} frames, "
                f"{metadata['pose_detection_rate']}% detected)"
            )
            return str(json_path)
        except OSError as exc:
            print(f"[Logger] WARNING: could not write session file — {exc}")
            return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def skipped_frames(self) -> int:
        """Frames where no valid side was available (side_used == 'none')."""
        return self._skipped_frames

    @property
    def warned_frames(self) -> int:
        """Frames that had at least one quality warning."""
        return self._warned_frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_metadata(self, ended_at: datetime, duration_s: float) -> Dict[str, Any]:
        total_frames = len(self._frames)
        detected_count = sum(1 for f in self._frames if f.get("pose_detected", False))

        proc_times = [
            f["processing_time_ms"]
            for f in self._frames
            if "processing_time_ms" in f
        ]

        # Derive per-frame FPS from successive timestamps
        timestamps = [f["timestamp_ms"] for f in self._frames]
        fps_mean = fps_median = fps_min = fps_max = 0.0
        if len(timestamps) >= 2:
            intervals = [
                timestamps[i + 1] - timestamps[i]
                for i in range(len(timestamps) - 1)
                if timestamps[i + 1] > timestamps[i]
            ]
            if intervals:
                fps_values = [1000.0 / iv for iv in intervals]
                fps_mean   = round(statistics.mean(fps_values), 1)
                fps_median = round(statistics.median(fps_values), 1)
                fps_min    = round(min(fps_values), 1)
                fps_max    = round(max(fps_values), 1)

        # Count side-fallback usage for bilateral exercises
        side_counts: Dict[str, int] = {"both": 0, "left": 0, "right": 0, "none": 0}
        for f in self._frames:
            side = f.get("side_used", "none")
            side_counts[side] = side_counts.get(side, 0) + 1

        total_reps = self._frames[-1].get("rep_count", 0)

        return {
            "session_id":               self._session_id,
            "exercise":                 self._exercise_name,
            "started_at":               self._started_at.isoformat(),
            "ended_at":                 ended_at.isoformat(),
            "duration_seconds":         round(duration_s, 2),
            "camera_resolution":        self._camera_resolution,
            "total_frames":             total_frames,
            "total_reps":               total_reps,
            "pose_detection_rate":      round(detected_count / max(total_frames, 1) * 100, 1),
            "fps_mean":                 fps_mean,
            "fps_median":               fps_median,
            "fps_min":                  fps_min,
            "fps_max":                  fps_max,
            "processing_time_mean_ms":  round(statistics.mean(proc_times), 1) if proc_times else 0.0,
            "side_both_pct":            round(side_counts["both"]  / max(total_frames, 1) * 100, 1),
            "side_left_pct":            round(side_counts["left"]  / max(total_frames, 1) * 100, 1),
            "side_right_pct":           round(side_counts["right"] / max(total_frames, 1) * 100, 1),
            "side_skipped_pct":         round(side_counts["none"]  / max(total_frames, 1) * 100, 1),
            "quality_warned_pct":       round(self._warned_frames  / max(total_frames, 1) * 100, 1),
        }

    def _write_json(self, metadata: Dict[str, Any]) -> Path:
        safe_name = self._exercise_name.replace(" ", "_").replace("-", "_")
        path = self._log_dir / f"{self._session_id}_{safe_name}.json"
        with open(path, "w") as fh:
            json.dump({"metadata": metadata, "frames": self._frames}, fh, indent=2)
        return path

    def _append_csv_summary(self, metadata: Dict[str, Any]) -> None:
        csv_path = self._log_dir / "sessions_summary.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metadata.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metadata)
