"""Background thread that snapshots+resets the metrics window on a fixed interval."""

import threading

from common.config import WINDOW_SECONDS


class WindowWorker(threading.Thread):
    def __init__(self, collector, interval_seconds: float = WINDOW_SECONDS):
        super().__init__(daemon=True)
        self.collector = collector
        self.interval = interval_seconds
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            self.collector.snapshot_window()

    def stop(self):
        self._stop.set()
