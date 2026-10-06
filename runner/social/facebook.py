"""Facebook Page video upload — draft first, publish only on explicit approval.

Uses the long-lived Page Access Token in .credentials/meta.env (no refresh
logic needed, see CLAUDE.md). Direct multipart upload from a local file —
Facebook, unlike Instagram, needs no public hosting.
"""

from __future__ import annotations

from pathlib import Path

import requests

from .env import load_env

GRAPH = "https://graph.facebook.com/v21.0"


class FacebookError(RuntimeError):
    pass


def _creds(root: Path) -> dict[str, str]:
    return load_env(root / ".credentials" / "meta.env")


def upload_draft(root: Path, video_path: Path, title: str, description: str) -> str:
    """Create an unpublished draft video post on the Page. Returns the
    video/post id. Reviewable in Meta Business Suite; not visible to the
    public until publish() is called. title and description are genuinely
    separate fields on this endpoint (confirmed against Meta's live docs -
    not assumed)."""
    c = _creds(root)
    with video_path.open("rb") as fh:
        r = requests.post(
            f"{GRAPH}/{c['META_PAGE_ID']}/videos",
            data={
                "access_token": c["META_PAGE_ACCESS_TOKEN"],
                "published": "false",
                "title": title,
                "description": description,
            },
            files={"source": (video_path.name, fh, "video/mp4")},
            timeout=600,
        )
    body = r.json()
    if "error" in body:
        raise FacebookError(f"upload_draft failed: {body['error']}")
    return body["id"]


def video_url(root: Path, video_id: str) -> str:
    """Permalink for a Page video, draft or published - Page admins can
    open this while logged in to preview an unpublished draft before
    approving it, the same role YouTube's private-video-viewable-by-owner
    link used to play (see discord_bot.py). NOT independently verified
    against a real unpublished draft the way the rest of this module's
    behavior has been - Meta's documented permalink format for a Page
    video, but confirm it actually renders before relying on it."""
    c = _creds(root)
    return f"https://www.facebook.com/{c['META_PAGE_ID']}/videos/{video_id}"


def publish(root: Path, video_id: str) -> None:
    """Flip a previously-created draft video to published (live on the
    Page). Only ever call this in response to explicit Discord approval."""
    c = _creds(root)
    r = requests.post(
        f"{GRAPH}/{video_id}",
        data={"access_token": c["META_PAGE_ACCESS_TOKEN"], "published": "true"},
        timeout=60,
    )
    body = r.json()
    if "error" in body:
        raise FacebookError(f"publish failed: {body['error']}")
