"""
Lucidscapes MCP server for Claude Code.

An "MCP server" is just a small program that gives Claude Code a set of buttons
it can press on your machine. Without it, Claude can only talk. With it, Claude
can check whether ComfyUI is alive, start a batch, read your log, and stop a run.

This is deliberately small. It does NOT run the pipeline itself — the pipeline
is runner/pipeline.py, which runs on its own with no Claude involved. This
server is only for when you are sitting at the desk and want to ask questions
or kick something off.

Register it once:
    claude mcp add lucidscapes -- python C:/path/to/lucidscapes/mcp/lucid_mcp.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from runner.comfy_client import ComfyClient  # noqa: E402
from runner.pipeline import Pipeline  # noqa: E402

app = FastMCP("lucidscapes")


@app.tool()
def status() -> str:
    """Is ComfyUI running, how hot is the GPU, how much VRAM is free, how many
    prompts are left to process, and where the finished videos are going."""
    return json.dumps(Pipeline().status(), indent=2)


@app.tool()
def check_setup() -> str:
    """Verify the whole configuration before an overnight run: every workflow
    file exists, is in API format, and every node title and setting named in
    config.yaml really exists in those workflows. Run this after ANY change to
    a workflow."""
    problems = Pipeline().check()
    return "Everything lines up." if not problems else "Problems:\n" + "\n".join("- " + p for p in problems)


@app.tool()
def inspect_workflow(filename: str) -> str:
    """List every node in a workflow with its type, its title, and its settings.
    Use this to find the exact title and setting name to put in config.yaml,
    for example when the SeedVR2 video loader wants 'video' vs 'file'."""
    return ComfyClient.describe(ComfyClient.load_workflow(ROOT / "workflows" / filename))


@app.tool()
def start_run(limit: int = 0) -> str:
    """Start the pipeline in the background so this chat is not blocked.
    limit=0 means process every unfinished prompt in the prompts file."""
    args = [sys.executable, "-m", "runner.pipeline", "run"]
    if limit:
        args += ["--limit", str(limit)]
    log = ROOT / "output" / "run.out"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        subprocess.Popen(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return f"Started in the background. Watch it with tail_log, or open {log}"


@app.tool()
def tail_log(lines: int = 40) -> str:
    """The last N lines of the run log — what stage each prompt is on, how long
    each one took, and anything that failed."""
    log = ROOT / "output" / "run.log"
    if not log.exists():
        return "No log yet."
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@app.tool()
def last_errors(lines: int = 60) -> str:
    """The last part of the error log, with full detail on recent failures."""
    log = ROOT / "output" / "errors.log"
    if not log.exists():
        return "No errors recorded."
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@app.tool()
def add_prompts(text: str) -> str:
    """Append new prompt blocks to prompts/prompts.txt.

    Pass the same format the file uses: IMAGE: and optional VIDEO: lines, with
    blocks separated by a line of three dashes. Useful when Claude has just
    written you a batch and you want them saved without copy-pasting."""
    path = ROOT / "prompts" / "prompts.txt"
    body = text.strip()
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n---\n\n" + body + "\n")
    from runner.pipeline import read_prompts
    return f"Appended. The file now holds {len(read_prompts(path, ''))} prompts."


@app.tool()
def list_finished(limit: int = 20) -> str:
    """The most recently finished videos in the FINAL folder, newest first."""
    pipe = Pipeline()
    folder = pipe.out_root / pipe.cfg["paths"]["final_folder"]
    if not folder.exists():
        return "Nothing finished yet."
    files = sorted(folder.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]
    return "\n".join(f"{f.name}  {f.stat().st_size / 1e6:.1f} MB" for f in files) or "empty"


@app.tool()
def stop_run() -> str:
    """Create the STOP file. The pipeline checks for it before each prompt and
    exits cleanly, without corrupting anything. Already-finished videos are
    kept and will not be redone."""
    (ROOT / "STOP").write_text("stop")
    return "STOP created. The run will halt after the current prompt finishes."


@app.tool()
def resume_run() -> str:
    """Delete the STOP file so runs can start again."""
    (ROOT / "STOP").unlink(missing_ok=True)
    return "STOP cleared."


@app.tool()
def forget_job(search: str) -> str:
    """Delete ledger entries whose generated image_prompt or video_prompt
    contains this text. Every job invents its own prompts via QwenVL now, so
    there's no fixed prompt to "redo" — this only trims history/stats, it
    does not queue anything to be regenerated."""
    pipe = Pipeline()
    hits = [k for k, v in pipe.ledger.data.items()
           if search.lower() in v.get("image_prompt", "").lower()
           or search.lower() in v.get("video_prompt", "").lower()]
    for k in hits:
        pipe.ledger.data.pop(k)
    pipe.ledger.path.write_text(json.dumps(pipe.ledger.data, indent=2), encoding="utf-8")
    return f"Removed {len(hits)} ledger entries."


if __name__ == "__main__":
    app.run()
