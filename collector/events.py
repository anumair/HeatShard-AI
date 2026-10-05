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


def write_events(events, path=None):
    """Overwrite the event-metadata feed. Scenarios call this to register
    events (e.g. an upcoming flash sale) before the corresponding traffic
    spike actually happens."""
    path = Path(path) if path else DEFAULT_EVENTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(events, f, indent=2)


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

    def visible_for_record(self, record_id: str, now: float):
        """Events for `record_id` that had already been announced at `now`.

        Replaying a recorded scenario loads the final event file, which
        contains events that were registered partway through the run; an
        event with an `announced_at` timestamp must stay invisible before
        that instant or the replay "knows the future". Events without the
        field (older feeds) are treated as always known."""
        return [e for e in self.for_record(record_id) if e.get("announced_at", float("-inf")) <= now]

    def watched_records(self, now: float, horizon_seconds: float, grace_seconds: float = 30.0) -> set:
        """Records with an already-announced event starting within
        `horizon_seconds` (or started less than `grace_seconds` ago) --
        the ones worth scoring even while they have no traffic yet, which
        is the whole point of the event channel's cold-start handling."""
        return {
            e["record_id"]
            for e in self._events
            if e.get("announced_at", float("-inf")) <= now and -grace_seconds <= e["scheduled_time"] - now <= horizon_seconds
        }

    def upcoming(self, horizon_seconds: float, now: float = None):
        """Events scheduled within [now, now + horizon_seconds]."""
        now = now if now is not None else time.time()
        return [e for e in self._events if 0 <= (e["scheduled_time"] - now) <= horizon_seconds]
