"""One-time interactive YouTube OAuth login.

Run this yourself at a real desktop with a browser:

    python -m runner.social.youtube_auth_setup

It opens a browser window, you sign in and grant upload access, and it
saves .credentials/youtube_token.json. Nothing else in this project can
complete this step - it needs a human physically clicking through Google's
consent screen. Re-run any time to refresh from scratch.
"""

from pathlib import Path

from . import youtube

ROOT = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    youtube.run_oauth_setup(ROOT)
