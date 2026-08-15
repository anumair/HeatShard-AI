"""Event-metadata channel: a pre-populated feed of scheduled demand events.

Lets the system anticipate load for records with no prior access history
(cold-start handling) -- e.g. a flash sale announced ahead of time. Used
as an input feature (EventSignal) starting in Stage 3's heat index and
actively exercised by Stage 2's flash-sale injection.
"""

import json
import time
from pathlib import Path

DEFAULT_EVENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "events.json"


class EventStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_EVENTS_PATH
        self._events = self._load()

    def _load(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return json.load(f)

    def reload(self):
        self._events = self._load()

    def all(self):
        return list(self._events)

    def for_record(self, record_id: str):
        return [e for e in self._events if e["record_id"] == record_id]

    def upcoming(self, horizon_seconds: float, now: float = None):
        """Events scheduled within [now, now + horizon_seconds]."""
        now = now if now is not None else time.time()
        return [e for e in self._events if 0 <= (e["scheduled_time"] - now) <= horizon_seconds]
