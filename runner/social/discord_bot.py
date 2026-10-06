"""Discord approval messages via the bot token — one credential, one code
path for both notifying and reading back the approve/reject reaction (a
plain webhook can only send, not read reactions back). See CLAUDE.md's
"Discord approval mechanism" section.
"""

from __future__ import annotations

from pathlib import Path

import requests

from .env import load_env

API = "https://discord.com/api/v10"
CHECK_EMOJI = "%E2%9C%85"   # ✅
CROSS_EMOJI = "%E2%9D%8C"   # ❌


class DiscordError(RuntimeError):
    pass


def _creds(root: Path) -> dict[str, str]:
    return load_env(root / ".credentials" / "discord.env")


def _headers(root: Path) -> dict[str, str]:
    return {"Authorization": f"Bot {_creds(root)['DISCORD_BOT_TOKEN']}"}


def get_bot_user_id(root: Path) -> str:
    """Fetch once per check_approvals.py run, not once per message — used
    to tell a real human reaction apart from the bot's own pre-react."""
    r = requests.get(f"{API}/users/@me", headers=_headers(root), timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def post_approval_message(root: Path, title: str, scene_text: str, facebook_preview_url: str) -> str:
    """One message per finished video. Posted immediately at staging time
    now (not deferred, unlike the old YouTube-gated flow this replaced) -
    the Facebook draft this links to is created synchronously right
    before this call, so there's no upload-completion wait to gate on.
    Page admins can open the link while logged in to preview the
    unpublished draft before approving. Returns the Discord message id.

    YouTube is paused entirely for now (channel banned) - see
    approvals.py::stage_job()."""
    channel_id = _creds(root)["DISCORD_CHANNEL_ID"]
    scene = (scene_text[:100] + "...") if len(scene_text) > 100 else scene_text
    lines = ["🎬 New video ready for review", ""]
    if title:
        lines.append(f"Title: {title}")
    lines.append(f"Scene: {scene}")
    lines.append("")
    lines.append(f"▶️ Preview on Facebook: {facebook_preview_url}")
    lines.append("")
    lines.append("React ✅ to publish everywhere · ❌ to discard")
    text = "\n".join(lines)
    r = requests.post(f"{API}/channels/{channel_id}/messages",
                      headers=_headers(root), json={"content": text}, timeout=30)
    if r.status_code >= 300:
        raise DiscordError(f"post_approval_message failed: {r.status_code} {r.text}")
    # No bot pre-react: it used to add both emoji itself so you could just
    # tap an existing reaction, but that made every fresh message show a
    # count of 1 on both ✅ and ❌ before anyone had actually voted -
    # confusing rather than helpful. check_reaction() never counted the
    # bot's own reaction as a vote anyway, so removing this changes nothing
    # about detection - the count now starts at 0 and only becomes 1 once
    # you react for real.
    return r.json()["id"]


def check_reaction(root: Path, message_id: str, bot_user_id: str) -> str:
    """Returns "approved", "rejected", or "pending" - "approved"/"rejected"
    only when a user OTHER than the bot itself has reacted, so the bot's
    own pre-react (a UX nicety, not a vote) never counts."""
    channel_id = _creds(root)["DISCORD_CHANNEL_ID"]
    for emoji, result in ((CHECK_EMOJI, "approved"), (CROSS_EMOJI, "rejected")):
        r = requests.get(
            f"{API}/channels/{channel_id}/messages/{message_id}/reactions/{emoji}",
            headers=_headers(root), timeout=30)
        if r.status_code >= 300:
            continue
        users = r.json()
        if any(u["id"] != bot_user_id for u in users):
            return result
    return "pending"
