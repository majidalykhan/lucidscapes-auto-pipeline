"""Periodic approval check — run every 10-15 min via Scheduled Task, same
pattern as the watchdog and disk/heat guards elsewhere in this project.

Each pass checks Discord reactions on every pending entry (Discord
messages are now posted immediately at staging time - see
approvals.py::stage_job() - there's no upload-completion wait to gate on
the way there used to be for YouTube). Approve -> fire the publish calls
for whatever platforms that job was staged to (only the ones not already
published - a partial failure on a previous pass must not re-publish
something that already succeeded), subject to rate_limit.py's daily cap
and minimum gap between publishes. Reject -> mark rejected.

YouTube is paused entirely - the channel was banned, and the working
theory is automated-posting behavior rather than content, which is also
why Facebook/Instagram publishing is now throttled here rather than fired
the instant a reaction is seen (see rate_limit.py's module docstring).

Never auto-publishes anything that hasn't had an explicit checkmark
reaction from a human. See CLAUDE.md's "Discord approval mechanism" and
"Idempotency" sections for the full design this implements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# output_root can be an absolute path on a different drive entirely (see
# config.yaml) - read the same config pipeline.py does, rather than
# assuming it's the project's own output/ folder, so this script's state
# files (pending_approvals.json, the YouTube queue) land wherever the
# generated media actually lives.
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
OUTPUT_ROOT = Path(_cfg["paths"]["output_root"])
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = ROOT / OUTPUT_ROOT

from .social import discord_bot, facebook, instagram, rate_limit  # noqa: E402
from .social.approvals import PendingApprovals  # noqa: E402


def log(msg: str) -> None:
    from datetime import datetime
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def maybe_complete(entry: dict) -> None:
    if all(entry.get(f"published_{p}") for p in entry["platforms"]):
        entry["status"] = "published"


def _publish_instagram(root: Path, output_root: Path, job_name: str,
                       entry: dict, state: PendingApprovals) -> None:
    """Create the container if none exists yet, publish it, and self-heal
    if the container turns out to have expired.

    A container can be stale for two reasons: an older version of this
    code used to create it immediately at generation time (see
    approvals.py::stage_job()'s docstring - that's what caused the first
    round of "Media expired" failures), or simply enough retries have
    passed that whatever container WAS created here has now aged out
    itself. Either way, Meta returns the same permanent, non-transient
    error (error_subcode 2207020) - there is nothing to wait out, the only
    fix is a fresh container. Rather than requiring a human to notice a
    stuck entry and manually clear its id (which is what happened the
    first time this surfaced), detect that specific error and do it here,
    automatically, retrying the publish once more in the same pass.

    Entries staged by that older code also never persisted video_path/
    ig_caption at all (there was nothing deferred to persist them for
    yet) - reconstruct both from what's still on disk instead of crashing
    on a missing key, the same backfill done by hand the first time this
    surfaced.
    """
    if not entry.get("video_path") or not entry.get("ig_caption"):
        entry["video_path"] = str(output_root / "FINAL" / f"video_{job_name}.mp4")
        gen_data_path = output_root / "generated_videos_data" / "generation_data.json"
        description = ""
        if gen_data_path.exists():
            gen_data = json.loads(gen_data_path.read_text(encoding="utf-8"))
            description = gen_data.get(job_name, {}).get("description", "")
        title = entry.get("title", "")
        entry["ig_caption"] = f"{title}\n\n{description}" if description else title
        state.save()

    if not entry.get("instagram_creation_id"):
        entry["instagram_creation_id"] = instagram.create_container(
            root, Path(entry["video_path"]), entry["ig_caption"])
        state.save()
    try:
        instagram.publish(root, entry["instagram_creation_id"])
    except instagram.InstagramError as e:
        if "2207020" not in str(e):
            raise
        log("instagram container had expired - creating a fresh one and retrying")
        entry["instagram_creation_id"] = instagram.create_container(
            root, Path(entry["video_path"]), entry["ig_caption"])
        state.save()
        instagram.publish(root, entry["instagram_creation_id"])


def check_pending(root: Path, output_root: Path, state: PendingApprovals, bot_user_id: str) -> None:
    for job_name, entry in state.pending_entries():
        try:
            result = discord_bot.check_reaction(root, entry["discord_message_id"], bot_user_id)
        except Exception as e:
            # One entry's Discord check failing (network blip, deleted
            # message, etc.) must not stop every other pending entry from
            # being checked this same pass.
            log(f"{job_name}: reaction check failed, will retry next pass: {e}")
            continue

        if result == "approved":
            for platform in entry["platforms"]:
                if entry.get(f"published_{platform}"):
                    continue
                allowed, reason = rate_limit.can_publish_now(output_root, platform)
                if not allowed:
                    log(f"{job_name}: {platform} publish deferred — {reason}")
                    continue
                try:
                    if platform == "facebook":
                        facebook.publish(root, entry["facebook_post_id"])
                        entry["published_facebook"] = True
                        rate_limit.record_publish(output_root, platform)
                        log(f"{job_name}: published to Facebook")
                    elif platform == "instagram":
                        # Container creation happens here, not at staging
                        # time - see approvals.py::stage_job()'s docstring
                        # for why (a creation_id expires long before a
                        # human typically reacts on Discord). _publish_instagram
                        # also self-heals a stale/expired id automatically -
                        # see its own docstring.
                        _publish_instagram(root, output_root, job_name, entry, state)
                        entry["published_instagram"] = True
                        rate_limit.record_publish(output_root, platform)
                        log(f"{job_name}: published to Instagram")
                except (facebook.FacebookError, instagram.InstagramError) as e:
                    # A failure on one platform must not stop the others,
                    # here or on other pending entries, and must not mark
                    # this platform published — next pass retries just this
                    # one, per the idempotency rule in CLAUDE.md.
                    log(f"{job_name}: publish to {platform} failed, will retry next pass: {e}")
            maybe_complete(entry)
            state.save()

        elif result == "rejected":
            entry["status"] = "rejected"
            log(f"{job_name}: rejected")
            state.save()


def main() -> None:
    # A missing state file means the output root is wrong or its drive is
    # unmounted - NOT "nothing pending". Silently treating it as empty is
    # what let an approved video sit unpublished with no error anywhere.
    state_file = OUTPUT_ROOT / "pending_approvals.json"
    if not state_file.exists():
        msg = f"{state_file} not found - output_root unreachable or wrong in config.yaml"
        log(f"ERROR: {msg}")
        try:
            OUTPUT_ROOT_FALLBACK = ROOT / "check_approvals_errors.log"
            with OUTPUT_ROOT_FALLBACK.open("a", encoding="utf-8") as f:
                f.write(f"[{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
        except Exception:
            pass
        sys.exit(2)
    state = PendingApprovals(OUTPUT_ROOT)
    if state.pending_entries():
        bot_user_id = discord_bot.get_bot_user_id(ROOT)
        check_pending(ROOT, OUTPUT_ROOT, state, bot_user_id)


if __name__ == "__main__":
    main()
