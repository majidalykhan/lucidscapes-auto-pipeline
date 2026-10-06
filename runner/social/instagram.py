"""Instagram Graph API: resumable upload direct from a local file.

Uses the modern upload_type=resumable container flow (confirmed against
Meta's current docs), NOT the older image_url/video_url flow the original
handoff doc assumed needed public hosting - no bucket, no public URL,
straight from disk. Requires "Facebook Login for Business" enabled on the
app and instagram_basic + instagram_content_publish permissions on the
Page token - both confirmed present on the regenerated token.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from .env import load_env

GRAPH = "https://graph.facebook.com/v21.0"
UPLOAD = "https://rupload.facebook.com/ig-api-upload"


class InstagramError(RuntimeError):
    pass


def _creds(root: Path) -> dict[str, str]:
    return load_env(root / ".credentials" / "meta.env")


def _ig_user_id(root: Path) -> str:
    c = _creds(root)
    r = requests.get(f"{GRAPH}/{c['META_PAGE_ID']}",
                     params={"fields": "instagram_business_account",
                             "access_token": c["META_PAGE_ACCESS_TOKEN"]}, timeout=30)
    body = r.json()
    account = body.get("instagram_business_account")
    if not account:
        raise InstagramError(f"no instagram_business_account linked to the Page: {body}")
    return account["id"]


def create_container(root: Path, video_path: Path, caption: str) -> str:
    """Create a resumable-upload container, upload the video binary
    directly from disk, and wait for Instagram to finish processing it.
    Returns the creation_id, ready for publish() below - does NOT publish
    anything itself. Posted as a Reel (REELS), the standard modern path
    for video content on the IG Graph API."""
    c = _creds(root)
    ig_user_id = _ig_user_id(root)

    r = requests.post(f"{GRAPH}/{ig_user_id}/media", data={
        "access_token": c["META_PAGE_ACCESS_TOKEN"],
        "media_type": "REELS",
        "upload_type": "resumable",
        "caption": caption,
    }, timeout=60)
    body = r.json()
    if "id" not in body:
        raise InstagramError(f"create_container failed: {body}")
    container_id = body["id"]

    file_size = video_path.stat().st_size
    with video_path.open("rb") as fh:
        upload = requests.post(
            f"{UPLOAD}/{container_id}",
            headers={
                "Authorization": f"OAuth {c['META_PAGE_ACCESS_TOKEN']}",
                "offset": "0",
                "file_size": str(file_size),
            },
            data=fh.read(),
            timeout=600,
        )
    if upload.status_code >= 300:
        raise InstagramError(f"binary upload failed: {upload.status_code} {upload.text}")

    _wait_until_finished(root, container_id)
    return container_id


def _wait_until_finished(root: Path, container_id: str, timeout: int = 300) -> None:
    c = _creds(root)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{GRAPH}/{container_id}",
                         params={"fields": "status_code,status",
                                 "access_token": c["META_PAGE_ACCESS_TOKEN"]}, timeout=30)
        body = r.json()
        status = body.get("status_code")
        if status == "FINISHED":
            return
        if status == "ERROR":
            raise InstagramError(f"container processing failed: {body}")
        time.sleep(5)
    raise InstagramError(f"container {container_id} did not finish processing within {timeout}s")


def publish(root: Path, creation_id: str) -> str:
    """Only ever call this in response to explicit Discord approval - there
    is no draft/private state at the API level once this is called, it
    goes live immediately. Returns the published media id."""
    c = _creds(root)
    ig_user_id = _ig_user_id(root)
    r = requests.post(f"{GRAPH}/{ig_user_id}/media_publish", data={
        "access_token": c["META_PAGE_ACCESS_TOKEN"],
        "creation_id": creation_id,
    }, timeout=60)
    body = r.json()
    if "id" not in body:
        raise InstagramError(f"publish failed: {body}")
    return body["id"]
