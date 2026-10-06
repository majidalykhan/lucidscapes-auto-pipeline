"""Publish-time rate limiting for Facebook and Instagram.

Why this exists: the user's working theory is that the YouTube channel got
banned specifically because of automated-posting behavior (uploads on a
tight, mechanical schedule, high daily volume, always via API with no
human usage pattern behind it) rather than the content itself. Removing
YouTube doesn't remove that risk - Facebook and Instagram are now the
only publish targets, and without some throttling they'd inherit the
exact same risk profile: draft creation already happens automatically at
generation time (unavoidable - that's what makes a Discord preview
possible at all), but the PUBLISH step (making something actually live
and public) is the part that's visible externally and worth throttling,
same as YouTube's own upload cap did for that platform.

This does NOT touch draft/container creation (see approvals.py /
check_approvals.py) - only the moment content actually goes live, which
is what a platform's abuse-detection systems would actually observe.

Same on-disk JSON pattern as youtube.py's daily counter (one file per
platform under output_root, {"date", "count"}), extended with a
last_published timestamp for the minimum-gap check.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

# Conservative starting points, same order of magnitude as YouTube's own
# now-unused cap - adjust once there's a real sense of what these accounts
# can sustain without tripping anything.
DAILY_PUBLISH_CAP = 12
MIN_PUBLISH_GAP_MINUTES = 20


def _state_path(output_root: Path, platform: str) -> Path:
    return output_root / f"{platform}_publish_state.json"


def _load(output_root: Path, platform: str) -> dict:
    path = _state_path(output_root, platform)
    if not path.exists():
        return {"date": "", "count": 0, "last_published": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(output_root: Path, platform: str, state: dict) -> None:
    path = _state_path(output_root, platform)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _today_count(state: dict) -> int:
    return state["count"] if state.get("date") == date.today().isoformat() else 0


def can_publish_now(output_root: Path, platform: str,
                     daily_cap: int = DAILY_PUBLISH_CAP,
                     min_gap_minutes: int = MIN_PUBLISH_GAP_MINUTES) -> tuple[bool, str | None]:
    """Returns (allowed, reason_if_not). Never raises - a missing/corrupt
    state file just means "nothing published yet today", not an error."""
    try:
        state = _load(output_root, platform)
    except Exception:
        state = {"date": "", "count": 0, "last_published": 0}
    today_count = _today_count(state)
    if today_count >= daily_cap:
        return False, f"daily {platform} publish cap ({daily_cap}) already reached today"
    elapsed_min = (time.time() - state.get("last_published", 0)) / 60
    if elapsed_min < min_gap_minutes:
        wait = round(min_gap_minutes - elapsed_min)
        return False, f"only {elapsed_min:.0f} min since last {platform} publish, need {min_gap_minutes} — {wait} min left"
    return True, None


def record_publish(output_root: Path, platform: str) -> None:
    """Call right after a publish call actually succeeds - bumps today's
    count and resets the gap timer."""
    state = _load(output_root, platform)
    today = date.today().isoformat()
    if state.get("date") != today:
        state = {"date": today, "count": 0, "last_published": 0}
    state["count"] += 1
    state["last_published"] = time.time()
    _save(output_root, platform, state)
