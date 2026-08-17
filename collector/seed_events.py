"""Regenerate data/events.json with scheduled_time relative to "now".

The checked-in sample is illustrative; run this before a demo so the
events actually sit in the near future relative to when you run the
checkpoint test / load generator.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.events import DEFAULT_EVENTS_PATH, write_events  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Seed the event-metadata feed")
    parser.add_argument("--record", action="append", default=[], help="record_id to schedule (repeatable)")
    parser.add_argument("--lead-seconds", type=float, default=60.0, help="how far in the future to schedule")
    parser.add_argument("--event-type", default="flash_sale")
    parser.add_argument("--magnitude", type=float, default=8.0)
    parser.add_argument("--out", default=str(DEFAULT_EVENTS_PATH))
    args = parser.parse_args()

    records = args.record or ["product:7", "product:23"]
    now = time.time()
    events = [
        {
            "record_id": record_id,
            "event_type": args.event_type,
            "scheduled_time": now + args.lead_seconds,
            "expected_magnitude": args.magnitude,
        }
        for record_id in records
    ]

    write_events(events, path=args.out)
    print(f"wrote {len(events)} event(s) to {args.out}, scheduled_time = now + {args.lead_seconds}s")


if __name__ == "__main__":
    main()
