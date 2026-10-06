"""One-click start/resume/stop for the Lucidscapes production run.

Run it whenever you want generation happening; stop it (Ctrl+C in this
window) whenever you want your GPU back for something else. Every run:

  1. Checks for a job that was cut off mid-generation (crash, power
     outage, forced shutdown - anything that killed the process before it
     could log a result) and rolls prompts_position.json back so that
     prompt gets regenerated instead of silently skipped. A job that
     failed cleanly (logged in the ledger) is NOT retried - that's the
     pipeline's existing, unchanged behavior; this only recovers a prompt
     that never reached any resolution at all.
  2. Makes sure ComfyUI is running, starting it if needed.
  3. Runs the pipeline until you stop it (or it hits the generous 24h
     safety cap in config.yaml).
  4. On the way out - however it ends - stops ComfyUI, so it's never left
     idling in the background using VRAM/RAM after this window closes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
OUT_ROOT = Path(CONFIG["paths"]["output_root"])
if not OUT_ROOT.is_absolute():
    OUT_ROOT = ROOT / OUT_ROOT
HOST = CONFIG["comfy"]["host"]
PORT = CONFIG["comfy"]["port"]
BASE_URL = f"http://{HOST}:{PORT}"
RESTART_CMD = CONFIG["run"]["comfy_restart_command"]
STOP_CMD = CONFIG["run"].get("comfy_stop_command")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def comfy_is_up() -> bool:
    try:
        urllib.request.urlopen(f"{BASE_URL}/system_stats", timeout=5)
        return True
    except Exception:
        return False


def recover_interrupted_job() -> None:
    """Roll prompts_position.json back if a job was mid-flight when this
    process (or the last one) died - see inflight_prompt.json's own
    docstring in pipeline.py for why a marker file is needed instead of
    just diffing the ledger (the ledger mixes in this whole project's
    entire history, and its failure entries don't even use the real
    job_name, so it can't tell "this exact prompt never resolved" from
    "totally unrelated job failed three days ago")."""
    marker = OUT_ROOT / "inflight_prompt.json"
    if not marker.exists():
        log("no interrupted job found - resuming cleanly")
        return
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"inflight marker exists but couldn't be read ({e}) - leaving position as-is")
        return
    position_file = OUT_ROOT / "prompts_position.json"
    try:
        position = json.loads(position_file.read_text(encoding="utf-8"))
    except Exception:
        position = {"next_index": 0}
    index_consumed = info.get("index_consumed")
    if index_consumed is None:
        marker.unlink(missing_ok=True)
        return
    if position.get("next_index", 0) > index_consumed:
        log(f"found a job cut off mid-generation ({info.get('job_name')}) - "
            f"rolling prompts_position.json back from {position.get('next_index')} "
            f"to {index_consumed} so that prompt gets regenerated")
        position_file.write_text(json.dumps({"next_index": index_consumed}), encoding="utf-8")
    marker.unlink(missing_ok=True)


def ensure_comfy_running() -> bool:
    if comfy_is_up():
        log("ComfyUI is already running")
        return True
    log("ComfyUI is not running - starting it")
    # cwd="C:/ComfyUI" matters: launching with this script's own cwd
    # inherited instead produced a ComfyUI process where Triton couldn't
    # find ptxas.exe and every QwenVL call failed - confirmed live
    # (relaunching the identical command with cwd fixed made an immediate
    # QwenVL test call succeed). comfy-cli apparently resolves some paths
    # relative to the caller's cwd rather than its own install root.
    subprocess.Popen(RESTART_CMD, shell=True, cwd="C:/ComfyUI")
    for _ in range(60):
        time.sleep(5)
        if comfy_is_up():
            log("ComfyUI is up - giving it 30s to finish loading nodes")
            time.sleep(30)
            return True
    log("ComfyUI did not come up after 5 minutes")
    return False


def tail_comfy_log(stop_event: threading.Event) -> None:
    """Stream ComfyUI's own console log into this window too.

    The pipeline's own run.log only summarizes progress per node ("running:
    SAMPLER", "step 4/8 (50%)") - it doesn't show which MODEL is loading or
    the finer-grained detail ComfyUI itself prints (SeedVR2's "Upscaling
    batch 18/48" / "Decoding batch 2/48", raw sampler iteration counters,
    etc.). This tails that file directly so both are visible together.

    Starts at the file's current end (only new lines from now on, same
    reasoning as every other "watch, don't replay history" tail in this
    project) and is logrotate-safe: if the file shrinks (ComfyUI restarted
    during a scheduled cooldown and started a fresh log), it just resets
    to the new file's start instead of erroring or going silent.
    """
    log_path = Path("C:/ComfyUI/user") / f"comfyui_{PORT}.log"
    pos = None
    while not stop_event.is_set():
        try:
            if not log_path.exists():
                time.sleep(2)
                continue
            size = log_path.stat().st_size
            if pos is None or size < pos:
                pos = size
            if size > pos:
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    new_data = f.read()
                    pos = f.tell()
                for line in new_data.splitlines():
                    if line.strip():
                        print(f"[ComfyUI] {line}", flush=True)
        except Exception:
            pass
        time.sleep(1)


def stop_comfy() -> None:
    if not STOP_CMD:
        return
    if not comfy_is_up():
        return
    log("stopping ComfyUI so your GPU is free")
    subprocess.run(STOP_CMD, shell=True)


def main() -> int:
    log("=== Lucidscapes production: starting ===")
    if not (OUT_ROOT / "ledger.json").exists():
        log(f"output root {OUT_ROOT} has no ledger.json - drive not mounted or "
            f"wrong path in config.yaml. Not starting.")
        return 1
    recover_interrupted_job()

    if not ensure_comfy_running():
        log("cannot continue without ComfyUI - exiting")
        return 1

    stop_tail = threading.Event()
    tail_thread = threading.Thread(target=tail_comfy_log, args=(stop_tail,), daemon=True)
    tail_thread.start()

    log("=== handing off to the pipeline - Ctrl+C any time to stop cleanly ===")
    proc = subprocess.Popen(
        [sys.executable, "-m", "runner.pipeline", "run", "--limit", "500"],
        cwd=str(ROOT),
    )
    try:
        return proc.wait()
    except KeyboardInterrupt:
        log("Ctrl+C received - stopping the pipeline")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 130
    finally:
        stop_tail.set()
        stop_comfy()
        log("=== stopped - GPU is free ===")


if __name__ == "__main__":
    sys.exit(main())
