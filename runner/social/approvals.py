"""Pending-approval state: <output_root>/pending_approvals.json.

Same job_name-keyed, plain-dict-on-disk pattern as pipeline.py's Ledger.
Each platform's publish state is tracked individually (not one combined
flag) so a partial failure only retries the platform that actually failed
- see CLAUDE.md's "Idempotency" note. `platforms` records which platforms
a given job was actually staged to, since Instagram is deferred right now
and a future job might not include it - "all platforms published" has to
mean all the ones that were actually staged, not a hardcoded three.

YouTube is paused entirely (the channel was banned - the working theory is
automated-posting behavior, not content, so removing it doesn't just fix
YouTube, it's also why check_approvals.py now rate-limits Facebook/
Instagram publishing - see runner/social/rate_limit.py). Re-adding YouTube
later just means restoring platforms to include it and reinstating
youtube.enqueue() below.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import discord_bot, facebook


class PendingApprovals:
    def __init__(self, output_root: Path):
        self.path = output_root / "pending_approvals.json"
        self.data: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def pending_entries(self):
        """Entries awaiting a Discord reaction check."""
        return [(k, v) for k, v in self.data.items()
                if v.get("status") == "pending" and v.get("discord_message_id")]


def stage_job(root: Path, output_root: Path, job_name: str, video_path: Path, scene_text: str,
              title: str, description: str, tags: list[str]) -> None:
    """Called once per finished video. Creates the Facebook draft
    immediately, then posts the Discord approval message right away too -
    unlike the old YouTube-gated flow, there's no upload to wait on, since
    the Facebook draft this links to already exists by the time this
    function returns. Never publishes anything.

    YouTube is paused entirely (channel banned - see rate_limit.py's
    module docstring for why that's not just a YouTube-specific problem).
    `tags` is kept in the signature unused rather than ripped out
    everywhere it's threaded through, so restoring YouTube later is a
    small, localized change instead of re-plumbing call sites.

    Instagram is deliberately NOT staged here - see instagram_creation_id
    left as None below. Confirmed live: a resumable-upload container's
    creation_id expires (Meta returns a permanent, non-transient "Media
    expired" / error_subcode 2207020 on media_publish) well before a human
    typically gets around to reacting on Discord, especially for an
    overnight batch reviewed the next morning. Facebook's draft has no
    such expiry, so only Instagram needs its creation deferred to the
    moment of actual approval (see check_approvals.py's instagram branch,
    which creates the container and publishes it back-to-back, right when
    the reaction is seen). video_path and the caption are stashed here so
    that later call has what it needs without regenerating anything.

    Each platform gets title/description routed to whatever fields it
    actually has (checked against Meta's live docs, not assumed):
    - Facebook: both title and description are set on the API (POST
      /{page_id}/videos has both as real separate fields), but Facebook's
      own field description for "description" is "text ... shown in a
      story about it" - the Feed-facing caption - while "title" is not
      prominently shown there at all (only in the video's own library/
      permalink view). Confirmed live: a viewer only ever saw the
      description text, never the title. So the description sent to
      Facebook leads with the title too, same as Instagram below, or the
      title would effectively never be seen.
    - Instagram: one caption field only, no separate title or tags at all -
      title and description are combined into it.
    - Neither platform has a usable public tags field (Facebook's
      content_tags needs numeric interest-graph IDs, custom_labels is
      internal-insights-only; Instagram has none at the API level).
    """
    state = PendingApprovals(output_root)

    fb_description = f"{title}\n\n{description}" if description else title
    facebook_post_id = facebook.upload_draft(root, video_path, title, fb_description)
    ig_caption = f"{title}\n\n{description}" if description else title

    discord_message_id = discord_bot.post_approval_message(
        root, title, scene_text, facebook.video_url(root, facebook_post_id))

    state.data[job_name] = {
        "platforms": ["facebook", "instagram"],
        "title": title,
        "scene_text": scene_text,
        "discord_message_id": discord_message_id,
        "facebook_post_id": facebook_post_id,
        "video_path": str(video_path),
        "ig_caption": ig_caption,
        "instagram_creation_id": None,
        "published_facebook": False,
        "published_instagram": False,
        "status": "pending",
    }
    state.save()
