"""YouTube Data API v3: one-time OAuth login, resumable upload, and a daily
upload cap so this project's 10-40 videos/night doesn't blow the API quota.

Quota mechanics: the real constraint is a per-channel UPLOAD COUNT limit,
not the raw 10,000/day quota-unit budget - unverified/new channels get
roughly 10-15 uploads/day, fully verified channels with advanced features
enabled up to ~100/day. (An earlier version of this file derived the cap
from quota units alone - 1,600 units/upload against the 10,000/day default
- which understated the real number; corrected here.) Either way the cap
applies to the UPLOAD itself, not to later flipping privacyStatus from
private to public (a cheap metadata update), so the daily limit has to
gate upload time, not publish time. Chosen strategy: spread uploads across
days - a small FIFO queue (youtube_upload_queue.json) holds jobs waiting
for their turn, drained at up to DAILY_UPLOAD_CAP per calendar day by
whatever process calls drain_queue() (check_approvals.py, on its normal
periodic schedule).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube"]
# NOT youtube.upload - confirmed live that scope covers uploading and basic
# snippet metadata (title/description/tags) but 403s with
# "insufficientPermissions" on videos.update(part="status", ...), which is
# exactly the call set_privacy() needs to flip private -> public. The
# broader scope also covers delete, mentioned as a possible future need in
# CLAUDE.md - one re-auth covers both gaps.
DAILY_UPLOAD_CAP = 12  # see module docstring - middle of the 10-15/day
                       # range for an unverified/new channel (this one is
                       # brand new); raise toward ~100 once the channel is
                       # fully verified with advanced features enabled.


class YouTubeError(RuntimeError):
    pass


def _paths(root: Path) -> tuple[Path, Path]:
    creds_dir = root / ".credentials"
    return creds_dir / "youtube_client_secret.json", creds_dir / "youtube_token.json"


def run_oauth_setup(root: Path) -> None:
    """One-time interactive login: opens a real browser window for the
    user to sign in and grant upload access, then saves youtube_token.json.
    Must be run by a human at a real desktop - this can't be completed
    from an automated/headless context. Safe to re-run; overwrites the
    token file with a fresh one.

    prompt="consent select_account" forces Google to show the account/
    channel picker rather than silently defaulting to whichever Google
    account is already signed in in the browser - without it, a first
    upload landed on the wrong channel (the main account instead of the
    Lucidscapes Brand Account) because the browser was already logged in
    and Google skipped the picker entirely.
    """
    client_secret, token_path = _paths(root)
    if not client_secret.exists():
        raise YouTubeError(f"missing {client_secret} - see CLAUDE.md credentials layout")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent select_account")
    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"[youtube] saved {token_path}")


def _credentials(root: Path) -> Credentials:
    _, token_path = _paths(root)
    if not token_path.exists():
        raise YouTubeError(
            "youtube_token.json does not exist yet - run "
            "`python -m runner.social.youtube_auth_setup` once, interactively, first."
        )
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload_video(root: Path, video_path: Path, title: str, description: str,
                  tags: list[str] | None = None, privacy_status: str = "private") -> str:
    """Resumable upload direct from a local file. Returns the video id.
    Costs ~1,600 quota units - callers must go through drain_queue() below
    rather than calling this directly from the main per-job flow, so the
    daily cap is actually respected."""
    creds = _credentials(root)
    youtube = build("youtube", "v3", credentials=creds)
    snippet = {"title": title, "description": description}
    if tags:
        snippet["tags"] = tags
    body = {
        "snippet": snippet,
        "status": {"privacyStatus": privacy_status},
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    return response["id"]


def set_privacy(root: Path, video_id: str, privacy_status: str) -> None:
    """Cheap metadata-only update - used to flip private -> public on
    approval. Does not count against the upload quota the way upload_video
    does."""
    creds = _credentials(root)
    youtube = build("youtube", "v3", credentials=creds)
    youtube.videos().update(
        part="status",
        body={"id": video_id, "status": {"privacyStatus": privacy_status}},
    ).execute()


# ---------------------------------------------------------------------------
# Daily upload queue - see module docstring.
#
# These take output_root separately from root (used above for
# .credentials/, which always stays with the project) - output_root is
# config.yaml's paths.output_root, which can point anywhere (e.g. a
# different drive when the project's own drive is low on space). Without
# this split, these state files would silently stay stranded on whatever
# drive the project lives on even after the generated media itself moved
# elsewhere.

def _queue_path(output_root: Path) -> Path:
    return output_root / "youtube_upload_queue.json"


def _counter_path(output_root: Path) -> Path:
    return output_root / "youtube_daily_count.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def enqueue(root: Path, output_root: Path, job_name: str, video_path: str, title: str,
            description: str, tags: list[str] | None = None) -> None:
    """Add a job to the upload queue instead of uploading immediately -
    called at staging time for every job, regardless of the daily cap."""
    queue = _load_json(_queue_path(output_root), [])
    queue.append({"job_name": job_name, "video_path": video_path,
                  "title": title, "description": description, "tags": tags or []})
    _save_json(_queue_path(output_root), queue)


def _today_count(output_root: Path) -> int:
    state = _load_json(_counter_path(output_root), {"date": "", "count": 0})
    if state.get("date") != date.today().isoformat():
        return 0
    return state.get("count", 0)


def _bump_today_count(output_root: Path) -> None:
    today = date.today().isoformat()
    state = _load_json(_counter_path(output_root), {"date": today, "count": 0})
    if state.get("date") != today:
        state = {"date": today, "count": 0}
    state["count"] += 1
    _save_json(_counter_path(output_root), state)


def cancel_queued(output_root: Path, job_name: str) -> bool:
    """Remove a not-yet-uploaded job from the queue - called on Discord
    rejection, so a rejected video never spends any of the daily quota.
    Returns False (no-op) if it already got uploaded before the rejection
    arrived; the caller doesn't need to do anything extra in that case,
    the already-uploaded private video just stays private."""
    queue = _load_json(_queue_path(output_root), [])
    kept = [item for item in queue if item["job_name"] != job_name]
    if len(kept) == len(queue):
        return False
    _save_json(_queue_path(output_root), kept)
    return True


def drain_queue(root: Path, output_root: Path) -> list[dict]:
    """Upload as many queued jobs as today's remaining cap allows, oldest
    first. Returns [{"job_name", "video_id"}] for whatever got uploaded
    this call, so the caller can update pending_approvals.json. Safe to
    call repeatedly (e.g. every check_approvals.py pass) - does nothing
    once today's cap or the queue itself is exhausted."""
    queue = _load_json(_queue_path(output_root), [])
    uploaded = []
    remaining_today = DAILY_UPLOAD_CAP - _today_count(output_root)
    while queue and remaining_today > 0:
        item = queue.pop(0)
        video_id = upload_video(root, Path(item["video_path"]), item["title"],
                                item["description"], tags=item.get("tags"),
                                privacy_status="private")
        _bump_today_count(output_root)
        remaining_today -= 1
        uploaded.append({"job_name": item["job_name"], "video_id": video_id})
    _save_json(_queue_path(output_root), queue)
    return uploaded
