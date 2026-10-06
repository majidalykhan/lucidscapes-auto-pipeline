"""
Lucidscapes overnight runner.

Reads a file of prompts and pushes each one through every stage in turn:

    prompt -> Flux still -> MiniMax video -> SeedVR2 -> RIFE -> final folder

One prompt at a time, one stage at a time, models unloaded in between. Nothing
runs in parallel, because you have one GPU and four large models, and running
them together is exactly how you get an out-of-memory crash at 3am.

Progress is written to a ledger file after every prompt, so if the machine
reboots you just start it again and it carries on where it stopped.

    python -m runner.pipeline run          # process every unfinished prompt
    python -m runner.pipeline run --limit 5
    python -m runner.pipeline status
    python -m runner.pipeline check        # verify config before a long run
    python -m runner.pipeline inspect 01_flux_still.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from .comfy_client import ComfyClient, ComfyError, OutOfMemory
from .social.approvals import stage_job as stage_social_upload

# Node titles can be anything the user typed in ComfyUI, emoji included (e.g.
# Video Helper Suite's own "Video Combine 🎥🅥🅗🅢") and they end up in stdout
# via progress logging, error messages, and check()'s problem list. Windows'
# default console codepage is cp1252, not UTF-8 — printing that title with no
# reconfiguration crashes with UnicodeEncodeError, taking the whole run down
# over what should have been a harmless log line. errors="replace" so a
# genuinely unencodable byte degrades the log line instead of the process.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
STOP_FILE = ROOT / "STOP"


# ---------------------------------------------------------------------------
# Camera direction, forced the same way as the scene brief below.
#
# Letting QwenVL pick a camera direction freely was tried first and failed
# the same way the scene subject did: confirmed in practice, 9 of 10 jobs
# in one batch came back "Slow pan left" regardless of what the instruction
# listed first or how the image looked. So Python forces the direction per
# job (build_scene_brief() below is the same pattern for the scene subject)
# and the video_prompt stage's instruction requires QwenVL to open with
# that exact phrase rather than choosing one itself — that forcing
# mechanism is unchanged by the LTX 2.5 switch. Only the wording changed,
# to match the terse "Locked cinematic shot, slow steady push-in toward
# the X..." style LTX responds best to (see config.yaml's video_prompt
# template): each phrase here is a sentence-opening fragment ending right
# before the subject noun, which QwenVL fills in from the actual image
# ("slow steady push-in toward" + "the cabin" + "."). The keyword half of
# each pair is what brief_adhered() substring-checks for.
CAMERA_MOVES = [
    ("slow steady pan left across", "pan left"),
    ("slow steady pan right across", "pan right"),
    ("slow steady pan up across", "pan up"),
    ("slow steady pan down across", "pan down"),
    ("slow steady push-in toward", "push-in"),
    ("slow steady pull-back from", "pull-back"),
]


# ---------------------------------------------------------------------------
# A small local QwenVL, given the same static instruction every job, tends
# to collapse onto whichever scene sits closest to its few-shot examples —
# confirmed in practice: several jobs in a row all came back "small stone
# cabin nestled in snowy ridgeline/hillside/cliffside," barely varying even
# with a fresh random seed each time and an instruction that explicitly
# said to vary the subject. Asking the model to invent its own variety
# isn't reliable; forcing it is. So Python randomly assembles a concrete
# brief — subject + weather + terrain + two accent details — once per job,
# and the 'prompt' stage's instruction requires QwenVL to build the scene
# around exactly that brief rather than choosing freely. This is what
# actually guarantees the variety the pipeline is supposed to produce.
# (phrase, keyword) — the keyword is what brief_adhered() checks for in
# QwenVL's actual output, so it must be a word QwenVL would plausibly write
# when genuinely describing that subject (kept lowercase, singular).
SCENE_SUBJECTS = [
    ("a small snow-dusted cabin", "cabin"),
    ("a cozy wooden hut", "hut"),
    ("a stone alpine chalet", "chalet"),
    ("a weathered fishing cottage", "cottage"),
    ("a small mountain lodge", "lodge"),
    ("a lit lighthouse", "lighthouse"),
    ("a stopped vintage train", "train"),
    ("a parked pickup truck", "truck"),
    ("a parked jeep", "jeep"),
    ("a small wooden sauna", "sauna"),
    ("a stone chapel", "chapel"),
    ("a lone watchtower", "watchtower"),
    ("a small farmhouse", "farmhouse"),
    ("an old mill house", "mill"),
    ("a cottage with a covered porch", "porch"),
]
SCENE_WEATHER = [
    "heavy falling snow", "light drifting snow", "cold steady rain",
    "thick rolling mist", "dense fog", "an overcast drizzle",
    "a clear freezing night with no precipitation", "a light blizzard",
    "morning frost with still air",
]
SCENE_TERRAIN = [
    "on a steep mountain ridge", "beside a frozen lake", "in a dense pine forest clearing",
    "on a coastal cliff above the sea", "in a wide valley", "beside a quiet river",
    "on a snowy mountain pass", "in an open snowy meadow", "on a rocky hillside",
]
SCENE_ACCENTS = [
    "a vehicle parked nearby with its headlights off and only a taillight glowing",
    "steam or smoke rising slowly from a chimney",
    "icicles hanging from the roofline",
    "a single lantern glowing beside the door",
    "frost patterns on the windows",
    "footprints leading up to the door",
]


def build_scene_brief() -> tuple[str, str]:
    """Returns (brief_text, subject_keyword). subject_keyword feeds
    brief_adhered() so a run_job() retry can tell a genuine attempt at the
    brief apart from the model falling back to its few-shot example."""
    subject_phrase, subject_keyword = random.choice(SCENE_SUBJECTS)
    weather = random.choice(SCENE_WEATHER)
    terrain = random.choice(SCENE_TERRAIN)
    accent_a, accent_b = random.sample(SCENE_ACCENTS, k=2)
    brief = f"{subject_phrase} {terrain}, {weather}, with {accent_a} and {accent_b}"
    return brief, subject_keyword


def brief_adhered(text: str, required) -> bool:
    """False almost always means the model fell back to reciting its
    few-shot example instead of building a scene around the given brief —
    confirmed in practice: 3 of 10 jobs in one batch came back with the
    example's exact wording ("small white wooden house perched on steep
    mountain ridge...") despite each having a completely different subject
    in their brief (a train, a lighthouse, a train). A same-seed retry
    would likely reproduce the same failure, so run_job() must also bump
    the seed when this returns False.

    `required` is either one keyword or an iterable of keywords, all of
    which must be present - the video_prompt stage checks both the forced
    camera-move phrase AND that it actually appended the TITLE:/etc.
    metadata lines rather than only writing the motion description."""
    keywords = [required] if isinstance(required, str) else list(required)
    return all(k.lower() in text.lower() for k in keywords)


def split_motion_and_metadata(text: str) -> tuple[str, str]:
    """Split the video_prompt stage's combined response into the motion
    description (everything before "TITLE:") and the metadata block
    (from "TITLE:" onward). Returns (motion_text, metadata_text) - the
    metadata_text is "" if the model never produced a TITLE: line at all
    (parse_metadata() on an empty string just yields empty defaults)."""
    match = re.search(r"TITLE\s*:", text, flags=re.IGNORECASE)
    if not match:
        return text.strip(), ""
    return text[:match.start()].strip(), text[match.start():].strip()


def parse_metadata(text: str) -> tuple[str, str, list[str]]:
    """Parse the metadata stage's TITLE:/DESCRIPTION:/TAGS: lines. Same
    line-prefix convention as read_prompts()'s IMAGE:/VIDEO: parsing.
    Falls back to empty values for anything missing rather than raising -
    a malformed metadata response should never take down an otherwise
    successful job."""
    title, description, tags = "", "", []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if upper.startswith("TITLE:"):
            title = line[len("TITLE:"):].strip()
        elif upper.startswith("DESCRIPTION:"):
            description = line[len("DESCRIPTION:"):].strip()
        elif upper.startswith("TAGS:"):
            tags = [t.strip() for t in line[len("TAGS:"):].split(",") if t.strip()]
    return title, description, tags


# ---------------------------------------------------------------------------
# prompts file
# ---------------------------------------------------------------------------

def read_prompts(path: Path, default_motion: str) -> list[dict]:
    """Read prompts/prompts.txt.

    Format — blocks separated by a line of three dashes:

        IMAGE: a lone freight train stopped on a snow-covered track ...
        VIDEO: locked shot, nothing moves, slow push-in, snow drifts
        ---
        IMAGE: a small cabin with one lit window ...

    VIDEO is optional. If you leave it out, the default motion line from
    config.yaml is used, which is what you want most of the time since your
    camera rules never change.

    Lines starting with # are ignored, so you can leave yourself notes.
    """
    raw = path.read_text(encoding="utf-8")
    prompts = []
    for block in raw.split("\n---"):
        image_parts, video_parts, current = [], [], None
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("IMAGE:"):
                current, line = image_parts, line[6:].strip()
            elif upper.startswith("VIDEO:"):
                current, line = video_parts, line[6:].strip()
            elif current is None:
                current = image_parts  # a bare block is treated as an image prompt
            if line:
                current.append(line)
        image = " ".join(image_parts).strip()
        if image:
            prompts.append({"image_prompt": image,
                            "video_prompt": " ".join(video_parts).strip() or default_motion})
    return prompts


def read_flux_prompts(path: Path) -> list[tuple[int | None, str]]:
    """Read a plain prompt list for the 'still' stage, as (id, prompt_text)
    pairs so the job-start log line can show which prompt number is in use
    - id is None for formats with no natural id field. Dispatches on file
    extension - .json for a structured {"prompts": [{"prompt": "..."}]}
    file (each entry already a complete, ready-to-use Flux prompt with
    style baked in), anything else for the older dash-separated plain-text
    format."""
    if path.suffix.lower() == ".json":
        return read_flux_prompts_json(path)
    return read_flux_prompts_dashed_text(path)


def read_flux_prompts_json(path: Path) -> list[tuple[int | None, str]]:
    """{"prompts": [{"id": 1, "category": "...", "prompt": "..."}, ...]} -
    in list order, id is only used for display (the job-start log line),
    the order in the file is what actually drives next_flux_prompt()'s
    position tracking, same as the text format below."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(entry.get("id"), entry["prompt"]) for entry in data["prompts"]]


def read_flux_prompts_dashed_text(path: Path) -> list[tuple[int | None, str]]:
    """One prompt per block, blocks separated by a line of 3+ dashes (any
    count - the file this was built for uses a 15-dash line, not
    read_prompts()'s 3-dash convention, and isn't in IMAGE:/VIDEO: format
    at all).

    A block occasionally carries one extra short line alongside the real
    prompt - e.g. a section header pasted in along with the surrounding
    text. Every genuine prompt here is one long comma-separated paragraph,
    so when a block has more than one non-blank line, only the longest one
    is kept and the stray line is dropped.
    """
    prompts: list[tuple[int | None, str]] = []
    current: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if re.fullmatch(r"-{3,}", line):
            if current:
                prompts.append((None, max(current, key=len)))
                current = []
            continue
        if line:
            current.append(line)
    if current:
        prompts.append((None, max(current, key=len)))
    return prompts


# ---------------------------------------------------------------------------
# ledger — a record of what's been generated, keyed by job, not by content
# ---------------------------------------------------------------------------
#
# Every job invents its own prompts via QwenVL, so there is no fixed prompt
# text to hash and no "have I already done this one?" question to ask —
# each run() call means "generate N more," full stop. The ledger is a
# history/stats log now, not a dedup gate. job_name (timestamp + a random
# suffix from uuid4, which — unlike Python's built-in hash() — never repeats
# across runs or reuses a value in a way that would collide) is already a
# unique identifier, so it's used directly as the ledger key.

class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def record(self, key: str, **info) -> None:
        self.data[key] = {"at": datetime.now().isoformat(timespec="seconds"), **info}
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def summary(self) -> dict:
        out = {}
        for v in self.data.values():
            out[v.get("status", "?")] = out.get(v.get("status", "?"), 0) + 1
        return out


# ---------------------------------------------------------------------------

def gpu_temp() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, config_path: Path = ROOT / "config.yaml"):
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.comfy = ComfyClient(**self.cfg["comfy"])
        self.run_cfg = self.cfg["run"]
        self.stages = self.cfg["stages"]
        self.out_root = ROOT / self.cfg["paths"]["output_root"]
        self.wf_dir = ROOT / self.cfg["paths"]["workflows"]
        # Always <output_root>/ledger.json, not a separately configurable
        # path - it used to be ("output/ledger.json" in paths.ledger),
        # which silently stopped following output_root the moment
        # output_root pointed somewhere else (e.g. a different drive),
        # leaving new runs writing history nobody would find next to the
        # actual generated media.
        # Refuse to start against an output root that has no history: a
        # missing/unmounted drive or a mistyped path would otherwise be
        # silently recreated empty, restarting prompts at index 0 and
        # splitting ledger/approval state across two folders. First-ever
        # setup: set LUCID_ALLOW_NEW_ROOT=1 once.
        if not (self.out_root / "ledger.json").exists() and \
                os.environ.get("LUCID_ALLOW_NEW_ROOT") != "1":
            raise SystemExit(
                f"output_root {self.out_root} has no ledger.json - drive not mounted or "
                f"wrong path in config.yaml. Fix it, or set LUCID_ALLOW_NEW_ROOT=1 to "
                f"deliberately start a fresh output folder.")
        self.ledger = Ledger(self.out_root / "ledger.json")
        self.deadline = datetime.now() + timedelta(hours=self.run_cfg["max_hours"])
        self.out_root.mkdir(parents=True, exist_ok=True)

        # Position within prompts.txt for any stage using `source:
        # prompts_file` (see next_flux_prompt()). Persisted so "take prompts
        # one by one until the end" spans multiple `run` invocations instead
        # of restarting at the top of the file every time.
        self._flux_prompts: list[str] | None = None
        self._flux_prompt_index = 0
        self._prompt_position_file = self.out_root / "prompts_position.json"
        self._inflight_marker = self.out_root / "inflight_prompt.json"
        if self._prompt_position_file.exists():
            try:
                self._flux_prompt_index = json.loads(
                    self._prompt_position_file.read_text(encoding="utf-8")
                ).get("next_index", 0)
            except Exception:
                pass

    # ---- audio shaping ----

    def shape_audio(self, path: Path, opts: dict) -> None:
        """Re-shape Stable Audio's output loudness in place (ffmpeg).

        The model bakes a "start loud, decay to silence" curve into
        whatever length it's asked for (measured: ~-13 dB in the first
        second down to -50..-87 dB by 8-10s, plus an occasional isolated
        loud hit), and asking for a longer clip only stretches the decay.
        So the stage generates a longer clip, and this keeps the first
        `keep_seconds`, flattens the level (dynaudnorm), then applies the
        wanted envelope: silent until fade_in_start, slow rise to
        fade_out_start, fade out to the end. Best-effort - on any failure
        the untouched original stays in place.
        """
        ff = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ff:
            self.log("    audio        shaping skipped: ffmpeg not on PATH")
            return
        keep = opts.get("keep_seconds", 10)
        fi_s, fi_d = opts.get("fade_in_start", 1), opts.get("fade_in_len", 7)
        fo_s, fo_d = opts.get("fade_out_start", 8), opts.get("fade_out_len", 2)
        floor = opts.get("fade_in_floor", 0.3)   # gain before/at the start of the swell
        # smoothstep swell from `floor` at fade_in_start up to 1.0 at
        # fade_in_start + fade_in_len (a plain afade-in would start at silence).
        swell = (f"if(lt(t,{fi_s}),{floor},if(lt(t,{fi_s + fi_d}),"
                 f"{floor}+(1-{floor})*pow((t-{fi_s})/{fi_d},2)*(3-2*(t-{fi_s})/{fi_d}),1))")
        chain = (f"atrim=0:{keep},asetpts=N/SR/TB,"
                 f"dynaudnorm=f=250:g=5:m=30:p=0.9:s=0,"
                 f"volume='{swell}':eval=frame")
        if fo_d > 0:   # fade_out_len 0 = no fade-out, sound runs to the last frame
            chain += f",afade=t=out:st={fo_s}:d={fo_d}:curve=qsin"
        tmp = path.with_name(path.stem + "_shaped.mp3")
        try:
            r = subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-af", chain,
                                "-c:a", "libmp3lame", "-q:a", "2", str(tmp)],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
                raise RuntimeError(r.stderr[-300:])
            tmp.replace(path)
            self.log(f"    audio        shaped: first {keep}s, soft start x{floor}, swell {fi_s}s->{fi_s + fi_d}s, "
                     + (f"fade out {fo_s}s->{fo_s + fo_d}s" if fo_d > 0 else "no fade-out"))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            self.log(f"    audio        shaping failed, keeping original: {e}")

    def grade_video(self, path: Path, job_name: str) -> None:
        """Tonal grade (ffmpeg) of the freshly rendered LTX clip, before upscale.

        Base grade, always on: a touch of black stretch, gamma lift and a
        saturation trim - keeps the dark, moody, unsaturated look while
        matching the still's brightness (the old fixed 1.2 contrast node
        crushed shadows; ComfyUI's LevelsAdjust node collapses a clip to
        one frame - hence ffmpeg, outside ComfyUI).

        Accent grade, limited run: while <output_root>/grade_accent.json has
        {"left": N > 0, ...}, ALSO boost saturation/contrast only where
        pixels are both bright and warm (windows, lamps, neon, fire) via a
        feathered mask, so the glow has life against the dark scene. The
        counter is decremented BEFORE rendering so a failure can't repeat it.
        The merge happens in RGB (gbrp): a grey mask on YUV leaks colour into
        the sky.
        """
        ff = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ff:
            self.log("    video        grade skipped: ffmpeg not on PATH")
            return
        b, g, sat = 0.012, 1.05, 0.86
        accent = None
        lift = ""          # phase 2: a little brightness/saturation, mainly highlights
        flag = self.out_root / "grade_accent.json"
        if flag.exists():
            try:
                o = json.loads(flag.read_text(encoding="utf-8"))
                if int(o.get("left", 0)) > 0:
                    accent = o
                    o["left"] = int(o["left"]) - 1
                    flag.write_text(json.dumps(o), encoding="utf-8")
                elif o.get("after"):
                    # The limited accent trial is over: from here on keep the
                    # warm accent AND add the small highlight-led lift.
                    accent = o
                    a2 = o["after"]
                    sat = a2.get("saturation", 1.04)
                    lift = (f":brightness={a2.get('brightness', 0.015)},"
                            f"curves=master='{a2.get('curve', '0/0 0.25/0.25 0.6/0.635 0.85/0.91 1/1')}'")
            except Exception:
                pass
        base = (f"colorlevels=rimin={b}:gimin={b}:bimin={b},"
                f"eq=gamma={g}:saturation={sat}{lift}")
        if accent:
            asat, acon = accent.get("saturation", 2.2), accent.get("contrast", 1.12)
            thr = accent.get("threshold", 70)
            m = (f"clip((0.2126*r(X,Y)+0.7152*g(X,Y)+0.0722*b(X,Y)-{thr})*3.6,0,255)"
                 "*clip((r(X,Y)-b(X,Y))/30,0,1)")
            fc = (f"[0:v]{base},format=gbrp,split=3[base][boost][m];"
                  f"[boost]eq=saturation={asat}:contrast={acon}[boosted];"
                  f"[m]geq=r='{m}':g='{m}':b='{m}',gblur=sigma=4[mask];"
                  f"[base][boosted][mask]maskedmerge,format=yuv420p[out]")
            vf = ["-filter_complex", fc, "-map", "[out]", "-map", "0:a?"]
        else:
            vf = ["-vf", base]
        sharpen = float(os.environ.get("LUCID_SHARPEN", "0.15") or 0)
        if sharpen > 0:   # after the grade, per the flow: grade -> sharpen -> upscale
            un = f"unsharp=5:5:{sharpen}:5:5:0"
            if accent:
                vf[1] = vf[1].replace("format=yuv420p[out]", f"format=yuv420p,{un}[out]")
            else:
                vf[1] = vf[1] + "," + un
        tmp = path.with_name(path.stem + "_graded.mp4")
        try:
            r = subprocess.run(
                [ff, "-v", "error", "-y", "-i", str(path), *vf,
                 "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p",
                 "-c:a", "copy", str(tmp)],
                capture_output=True, text=True, timeout=900)
            if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 10000:
                raise RuntimeError(r.stderr[-300:])
            tmp.replace(path)
            self.log(f"    video        graded (base{' + WARM-ACCENT test' if accent else ''}): {job_name}")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            self.log(f"    video        grade failed, keeping original: {e}")

    # ---- logging ----

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with (self.out_root / "run.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ---- guards ----

    def wait_if_hot(self) -> bool:
        """Pause while the GPU cools. Returns False if it never cools down."""
        limit = self.run_cfg["max_gpu_temp_c"]
        waited = 0
        while True:
            temp = gpu_temp()
            if temp is None or temp < limit:
                return True
            if waited >= self.run_cfg["max_cooldown_minutes"] * 60:
                self.log(f"GPU stuck at {temp}C after cooling for "
                         f"{self.run_cfg['max_cooldown_minutes']} min — stopping")
                return False
            self.log(f"GPU at {temp}C (limit {limit}C) — pausing 2 min to cool")
            time.sleep(120)
            waited += 120

    def blocked(self) -> str | None:
        if STOP_FILE.exists():
            return "STOP file found"
        if datetime.now() > self.deadline:
            return "time budget used up"
        free_gb = shutil.disk_usage(self.out_root).free / 1e9
        if free_gb < self.run_cfg["min_free_disk_gb"]:
            return f"only {free_gb:.0f} GB disk left"
        return None

    def ensure_comfy(self) -> bool:
        """Make sure ComfyUI is alive, restarting it if configured to."""
        if self.comfy.is_up():
            return True
        cmd = self.run_cfg.get("comfy_restart_command")
        if not cmd:
            self.log("ComfyUI is not responding and no restart command is set")
            return False
        self.log("ComfyUI is not responding — restarting it")
        # cwd matters here: launching this same command while inherited
        # cwd was this project's own folder (not ComfyUI's own workspace
        # root) produced a ComfyUI process where Triton couldn't find
        # ptxas.exe and every QwenVL call failed - confirmed by relaunching
        # the identical command with cwd fixed and a QwenVL call
        # immediately succeeding. comfy-cli apparently resolves some paths
        # relative to the caller's cwd rather than its own install root.
        subprocess.Popen(cmd, shell=True, cwd="C:/ComfyUI")
        for _ in range(60):
            time.sleep(5)
            if self.comfy.is_up():
                self.log("ComfyUI is back")
                return True
        self.log("ComfyUI did not come back")
        return False

    def periodic_cooldown(self) -> bool:
        """Scheduled break every `cooldown_every_n_jobs` jobs, independent of
        temperature - wait_if_hot() above is reactive (only triggers if the
        GPU is actually running hot); this is a preventive pause for a long
        unattended overnight batch, run unconditionally on a job count.

        Stopping ComfyUI outright (rather than just the per-stage /free
        call already used everywhere else) is deliberate: /free only
        unloads model weights from VRAM, not whatever a custom node may
        have accumulated in the Python process's own heap over many
        sequential jobs. A full stop guarantees both VRAM and RAM are
        genuinely empty for the whole cooldown window, not just "mostly"
        - the small restart overhead afterward is cheap next to a 15 min
        pause. Returns False if ComfyUI doesn't come back afterward, so
        the caller can stop the run instead of failing every job from here
        on.
        """
        minutes = self.run_cfg.get("cooldown_minutes", 15)
        self.log(f"=== scheduled cooldown: stopping ComfyUI, "
                 f"pausing {minutes} min to clear VRAM/RAM and let the GPU rest ===")
        stop_cmd = self.run_cfg.get("comfy_stop_command")
        if stop_cmd:
            subprocess.run(stop_cmd, shell=True)
        # A silent 15-minute sleep is indistinguishable from a hang to
        # anyone watching the console (it was reported as "stuck" the first
        # time this ran), so log a heartbeat every minute.
        for remaining in range(minutes, 0, -1):
            self.log(f"cooldown: {remaining} min left (ComfyUI is stopped on purpose)")
            time.sleep(60)
        if not self.ensure_comfy():
            return False
        # is_up() only confirms the HTTP server is bound, not that every
        # custom node has finished importing / the Manager's registry
        # fetch has settled - submitting a real job too soon after a fresh
        # launch has caused spurious stage timeouts before (see CLAUDE.md).
        # Cheap insurance given the alternative is losing a job for no
        # real reason after just having paused 15 minutes anyway.
        self.log("cooldown: ComfyUI is back up, giving it 30s to finish loading")
        time.sleep(30)
        self.log("=== cooldown complete, resuming ===")
        return True

    # ---- prompts.txt, for a `source: prompts_file` stage ----

    def next_flux_prompt(self) -> tuple[int | None, str] | None:
        """Pop the next unconsumed (id, prompt_text) pair from prompts.txt,
        in file order. Returns None once the file is exhausted."""
        if self._flux_prompts is None:
            self._flux_prompts = read_flux_prompts(ROOT / self.cfg["paths"]["prompts_file"])
        if self._flux_prompt_index >= len(self._flux_prompts):
            return None
        prompt = self._flux_prompts[self._flux_prompt_index]
        self._flux_prompt_index += 1
        self._prompt_position_file.write_text(
            json.dumps({"next_index": self._flux_prompt_index}), encoding="utf-8")
        return prompt

    def _write_generation_data(self, generation_id: str, flux_image_prompt: str, video_prompt: str,
                               title: str, description: str, tags: list[str]) -> None:
        """generated_videos_data/generation_data.json - the one persistent
        record for a generation, keyed by generation_id (job_name
        throughout this pipeline): the still-image prompt, the motion
        prompt, and the title/description/tags every platform's staging
        call uploads alongside the video. Replaces the old scattered
        per-job .txt files in 00_prompts/02b_video_prompts (now deleted
        right after being read, see run_job()) and the old
        videos_metadata.json - one id, one place to find everything about
        that generation. Written once the video itself exists (see the
        'video' stage in run_job()), same read-modify-write pattern as
        Ledger."""
        path = self.out_root / "generated_videos_data" / "generation_data.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data[generation_id] = {
            "generation_id": generation_id,
            "flux_image_prompt": flux_image_prompt,
            "video_prompt": video_prompt,
            "title": title,
            "description": description,
            "tags": tags,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---- one stage ----

    def run_stage(self, stage: dict, ctx: dict) -> Path:
        """Run a single workflow and return the file it produced.

        ctx carries values between stages: the prompts, the seed, and
        previous_file, which is whatever the last stage output.
        """
        graph = ComfyClient.load_workflow(self.wf_dir / stage["workflow"])

        for item in stage.get("set", []):
            source = item["from"]
            if source == "previous_file":
                value = self.comfy.stage_input_file(ctx["previous_file"])
            elif source == "audio_file":
                # Same file-staging as previous_file (copies into ComfyUI's
                # input folder, returns the "subfolder/name" a LoadAudio-
                # style COMBO widget expects) but reads the audio stage's
                # own output instead - see sets_file: audio_file above.
                value = self.comfy.stage_input_file(ctx["audio_file"])
            elif source == "seed":
                ComfyClient.set_seed(graph, item["title"], ctx["seed"])
                continue
            elif source == "literal":
                value = item["value"]
            elif source == "template":
                # Combine a fixed instruction with a ctx value using
                # {curly_braces} naming the ctx key directly (e.g.
                # {image_prompt}) — needed because QwenVL exposes only one
                # big text box, not separate system/user fields.
                value = item["template"].format(**ctx)
            else:
                value = ctx[source]

            if "json_field" in item:
                # Some custom nodes (Pixaroma's prompt and seed boxes are the
                # ones you'll hit) don't expose a plain 'text' or 'seed'
                # setting. Instead the whole widget is one blob holding several
                # fields, e.g. SeedState = '{"runSeed": 123}'. Read whatever is
                # there now, change only the one field named in json_field,
                # and write the rest back untouched.
                #
                # That blob is usually a STRING containing JSON text, not a
                # native object — you can tell from how it prints: a real
                # Python dict shows single quotes and a space after the colon
                # {'runSeed': 123}; double quotes with no space, like
                # {"runSeed":123}, is JSON text sitting inside a string. If we
                # handed a parsed dict back to ComfyUI instead of re-encoding
                # it to a string, the node would receive a different data type
                # than it declared, which can fail validation or, worse, be
                # silently ignored — exactly how a seed override can appear to
                # do nothing while the render keeps reusing the old value. So
                # this always writes back in the same form it was read in.
                node_id = ComfyClient.find_by_title(graph, item["title"])
                current = graph[node_id]["inputs"].get(item["key"], {})
                was_string = isinstance(current, str)
                blob = json.loads(current) if was_string else dict(current)
                blob[item["json_field"]] = value
                value = json.dumps(blob) if was_string else blob

            ComfyClient.set_input(graph, item["title"], item["key"], value)

        out_dir = self.out_root / stage["save_to"]
        text_stage = stage.get("produces") == "prompt"
        if text_stage:
            # "CR Save Text To File" writes straight to disk itself — no
            # os.makedirs on its side — rather than through a normal ComfyUI
            # save node, so nothing else creates this folder first.
            out_dir.mkdir(parents=True, exist_ok=True)
            # output_file_path is otherwise whatever was baked into the
            # workflow JSON at export time - fine as long as output_root
            # never changes, but the moment it does (e.g. moved to a
            # different drive, as happened this session) that baked-in
            # path silently stops matching out_dir and every job fails at
            # its very first stage with a FileNotFoundError from the save
            # node. Force it to the real out_dir every run instead of
            # trusting the export, using the same node title file_name is
            # already targeted on.
            file_name_item = next((i for i in stage["set"] if i.get("key") == "file_name"), None)
            if file_name_item:
                ComfyClient.set_input(graph, file_name_item["title"], "output_file_path", str(out_dir))

        stage_name = stage["name"]
        files = self.comfy.run(
            graph,
            timeout=stage["timeout_minutes"] * 60,
            show_progress=self.run_cfg.get("show_progress", True),
            on_update=lambda msg: self.log(f"    {stage_name:<12} {msg}"),
        )

        if text_stage:
            # This save node never reports its output through ComfyUI's
            # normal job-output API (confirmed by reading its source: it
            # returns a plain help-text string, no {"ui": ...} payload) — so
            # there is nothing to match against `files`. Its file_name is set
            # to job_name via config.yaml's `set:`, and its output_file_path
            # is baked into the workflow as exactly out_dir, so the path is
            # known outright rather than discovered from ComfyUI's response.
            dest = out_dir / f"{ctx['job_name']}{stage['extensions'][0]}"
            if not dest.exists():
                raise ComfyError(
                    f"stage '{stage['name']}' should have written {dest} (via "
                    f"its CR Save Text To File node) but the file isn't there. "
                    f"Check output_file_path/file_name on that node in "
                    f"{stage['workflow']}."
                )
            return dest

        wanted = tuple(stage["extensions"])
        matches = [f for f in files if f["filename"].lower().endswith(wanted)]
        if not matches:
            got = [f["filename"] for f in files] or ["nothing"]
            raise ComfyError(
                f"stage '{stage['name']}' produced {got}, but this stage expects "
                f"a {' or '.join(wanted)} file. Check the save node in "
                f"{stage['workflow']}, or widen 'extensions' in config.yaml."
            )

        dest = out_dir / f"{stage.get('file_prefix', '')}{ctx['job_name']}{Path(matches[-1]['filename']).suffix}"
        return self.comfy.download(matches[-1], dest)

    # ---- one job, all stages ----

    def run_job(self, index: int, total: int) -> bool:
        """Run one full job end to end: QwenVL invents the still prompt,
        Flux renders it, QwenVL looks at that still and writes the motion
        prompt, MiniMax animates it, then upscale and interpolate. Nothing
        here is supplied by a human — every prompt is generated fresh by the
        stages themselves, per CLAUDE.md's "no manual work" design.
        """
        seed = random.randint(1, 2**31 - 1) if self.run_cfg["randomize_seed"] else self.run_cfg["fixed_seed"]
        job_name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        scene_brief, scene_subject_keyword = build_scene_brief()
        camera_move, camera_move_keyword = random.choice(CAMERA_MOVES)
        ctx = {"job_name": job_name, "seed": seed,
               "image_prompt": "", "video_prompt": "",
               "video_title": "", "video_description": "", "video_tags": [],
               "scene_brief": scene_brief, "camera_move": camera_move,
               "previous_file": os.environ.get("LUCID_START_IMAGE") or None}   # test-only: start from an existing still
        # Which ctx key a produces:prompt stage sets, and the keyword(s)
        # its text must contain to count as actually having followed the
        # forced brief/camera-move rather than drifting off it.
        # video_prompt requires both the forced camera phrase AND the
        # TITLE: line - it's a single combined QwenVL call now: the model
        # looks at the still image once and writes the motion description
        # plus the upload title/description/tags in the same response
        # (originally a separate later "metadata" stage/call, folded in
        # here after finding that a 3rd sequential QwenVL call in the same
        # ComfyUI session was unreliably reproducing a PRIOR call's exact
        # output verbatim instead of processing its own instruction - see
        # git history/CLAUDE.md. Two QwenVL calls per job instead of three
        # sidesteps that rather than chasing the root cause further).
        required_keywords = {"image_prompt": scene_subject_keyword,
                             "video_prompt": (camera_move_keyword, "title:")}

        # scene_brief is only meaningful when the 'prompt' stage actually
        # builds image_prompt from it (QwenVL invention) - suppress it from
        # the log when that stage is instead sourced from prompts.txt, so
        # the line doesn't show a random brief that has nothing to do with
        # what's actually about to be generated.
        uses_prompts_file = any(s.get("source") == "prompts_file"
                                and s.get("sets", "image_prompt") == "image_prompt"
                                for s in self.stages)

        # Consumed here, ahead of the stage loop below, purely so the
        # job-start banner can show which prompt number this job is using
        # - the loop's own source:prompts_file branch then just uses this
        # same already-fetched value instead of pulling a second one.
        prefetched_prompt = None
        if uses_prompts_file:
            prefetched_prompt = self.next_flux_prompt()
            if prefetched_prompt is None:
                self.log("prompt       prompts.txt is exhausted — stopping the run")
                return False
        prompt_id = prefetched_prompt[0] if prefetched_prompt else None
        prompt_note = f"prompt #{prompt_id}  " if prompt_id is not None else ""

        brief_note = "" if uses_prompts_file else f"brief: {scene_brief}  "
        self.log(f"--- job {index}/{total}: {job_name}  "
                 f"{prompt_note}{brief_note}camera: {camera_move}")
        started = time.time()

        for stage in self.stages:
            if not stage.get("enabled", True):
                continue
            if stage.get("source") == "prompts_file":
                # Bypasses QwenVL and run_stage() entirely: this stage's
                # value comes straight from prompts.txt instead of being
                # invented, one prompt consumed per job in file order.
                sets_key = stage.get("sets", "image_prompt")
                prompt_text = prefetched_prompt[1]
                ctx[sets_key] = prompt_text
                self.log(f"    {stage['name']:<12} from prompts.txt -> "
                         f"{sets_key}: {prompt_text[:70]}...")
                # A crash mid-job (power outage, process kill) skips both
                # run_job()'s own return and run()'s per-job except block
                # below, so nothing would normally record that this
                # specific prompt was consumed but never finished - the
                # position file alone can't distinguish "job completed
                # cleanly" from "job was in flight when everything died".
                # This marker exists for exactly that gap: written the
                # moment a prompt is consumed, cleared the moment this job
                # reaches either a normal return or run()'s except handler.
                # If it's still there at the next startup, resume_production.py
                # rolls prompts_position.json back so that prompt gets
                # regenerated instead of silently skipped.
                self._inflight_marker.write_text(
                    json.dumps({"job_name": job_name,
                                "index_consumed": self._flux_prompt_index - 1}),
                    encoding="utf-8")
                continue
            if not self.wait_if_hot():
                return False
            if not self.ensure_comfy():
                return False

            attempts = self.run_cfg["oom_retries"] + 1
            for attempt in range(attempts):
                try:
                    t0 = time.time()
                    result = self.run_stage(stage, ctx)
                    if stage.get("produces") == "prompt":
                        # Text stage: its content overwrites the named ctx
                        # key (image_prompt or video_prompt), never
                        # previous_file — that must keep pointing at the
                        # last real media file so the next stage's
                        # LOAD_IMAGE/LOAD_VIDEO still gets it, not a .txt.
                        text = Path(result).read_text(encoding="utf-8", errors="replace").strip()
                        required = required_keywords.get(stage.get("sets"))
                        if required and not brief_adhered(text, required):
                            # QwenVL ignored the forced brief/camera-move (or,
                            # for video_prompt, skipped the TITLE:/etc. lines
                            # entirely) and fell back to its few-shot example
                            # instead — a same-seed retry would likely repeat
                            # it, so bump the seed (also reused by every later
                            # stage, which is fine — it's a per-job value)
                            # and try this one stage again, once.
                            missing = required if isinstance(required, str) else \
                                ", ".join(k for k in required if k.lower() not in text.lower())
                            self.log(f"    {stage['name']:<12} ignored the required phrasing "
                                     f"(missing: {missing}) "
                                     f"— retrying with a fresh seed")
                            ctx["seed"] = random.randint(1, 2**31 - 1)
                            # The first attempt's file is still sitting at
                            # the exact path run_stage() predicts for the
                            # retry too (job_name is unchanged) - CR Save
                            # Text To File never overwrites, it renames to
                            # _1/_2/... on collision instead. Without
                            # deleting it first, run_stage() would keep
                            # returning this SAME stale first-attempt path,
                            # silently re-reading its old (non-adherent)
                            # text forever instead of the retry's real
                            # output, which would land in a _1 file no one
                            # ever looks at. Confirmed happening in
                            # practice via leftover _1.txt files on disk.
                            Path(result).unlink(missing_ok=True)
                            result = self.run_stage(stage, ctx)
                            text = Path(result).read_text(encoding="utf-8", errors="replace").strip()
                            if not brief_adhered(text, required):
                                self.log(f"    {stage['name']:<12} still ignored it "
                                         f"after retry — continuing anyway")
                        if stage["name"] == "video_prompt":
                            # Combined response: motion description, then
                            # the TITLE:/DESCRIPTION:/TAGS: metadata lines.
                            # ctx["video_prompt"] must stay JUST the motion
                            # part — MiniMax reads it directly as its prompt
                            # and would otherwise get the metadata text too.
                            motion_text, metadata_text = split_motion_and_metadata(text)
                            ctx["video_prompt"] = motion_text
                            ctx["video_title"], ctx["video_description"], ctx["video_tags"] = \
                                parse_metadata(metadata_text)
                            transcript_text = motion_text
                        else:
                            ctx[stage["sets"]] = text
                        # This stage's own scratch .txt (needed only so the
                        # ComfyUI save node had somewhere to write, then
                        # read back above) is not a persistent artifact -
                        # everything worth keeping lives in one place now,
                        # generated_videos_data/generation_data.json,
                        # written once the job finishes. Deleting this
                        # immediately is what actually stops 00_prompts/
                        # and 02b_video_prompts/ from accumulating one file
                        # per job forever.
                        Path(result).unlink(missing_ok=True)
                        self.log(f"    {stage['name']:<12} ok in {time.time() - t0:>5.0f}s  "
                                 f"-> {stage['sets']}: {text[:70]}...")
                    else:
                        # Most media stages feed the next one via
                        # previous_file (still -> video -> upscale ->
                        # interpolate all chain this way). The audio stage
                        # is the odd one out: it also consumes
                        # previous_file (the LTX video, to look at and
                        # generate a music prompt from) but its OWN output
                        # is an audio file, not the next video in the
                        # chain - sets_file: audio_file routes it into a
                        # separate ctx key instead, so previous_file still
                        # points at the LTX video for the upscale stage
                        # right after it.
                        dest_key = stage.get("sets_file", "previous_file")
                        if stage["name"] == "video":
                            self.grade_video(Path(result), job_name)
                        if stage.get("shape_audio"):
                            self.shape_audio(Path(result), stage["shape_audio"])
                        ctx[dest_key] = result
                        self.log(f"    {stage['name']:<12} ok in {time.time() - t0:>5.0f}s  "
                                 f"-> {Path(result).name}")
                        if stage["name"] == "video":
                            # The video itself now exists - write the one
                            # consolidated record for this generation
                            # (image prompt, motion prompt, title/
                            # description/tags - all parsed earlier).
                            self._write_generation_data(
                                job_name, ctx["image_prompt"], ctx["video_prompt"],
                                ctx["video_title"], ctx["video_description"], ctx["video_tags"])
                    break
                except OutOfMemory:
                    # Recoverable. Drop everything from VRAM, breathe, try again.
                    self.log(f"    {stage['name']}: out of memory "
                             f"(attempt {attempt + 1}/{attempts}) — unloading models")
                    self.comfy.interrupt()
                    self.comfy.free_memory()
                    time.sleep(self.run_cfg["oom_wait_seconds"])
                    if attempt + 1 == attempts:
                        raise
                except ComfyError:
                    raise

            if self.run_cfg["free_vram_between_stages"]:
                self.comfy.free_memory()
                time.sleep(self.run_cfg["unload_wait_seconds"])

        final = self.out_root / self.cfg["paths"]["final_folder"] / Path(ctx["previous_file"]).name
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ctx["previous_file"], final)

        mins = (time.time() - started) / 60
        self.log(f"    DONE in {mins:.1f} min -> {final}")
        self.ledger.record(job_name, status="done", file=str(final), seed=seed,
                           minutes=round(mins, 1), image_prompt=ctx["image_prompt"][:400],
                           video_prompt=ctx["video_prompt"][:400])

        try:
            fallback = ctx["video_prompt"][:400] or ctx["image_prompt"][:400]
            stage_social_upload(
                ROOT, self.out_root, job_name, final,
                scene_text=ctx["image_prompt"],
                title=ctx["video_title"] or job_name,
                description=ctx["video_description"] or fallback,
                tags=ctx["video_tags"],
            )
            self.log("    social       staged on YouTube (queued)/Facebook/Instagram; "
                     "Discord review post follows once the YouTube upload completes")
        except Exception as e:
            # The video itself is done and safely in FINAL/ regardless of
            # this - a social-staging failure must never look like the
            # whole job failed, and must never lose or re-do the video.
            self.log(f"    social       staging failed, video is still in FINAL/: {e}")

        self._inflight_marker.unlink(missing_ok=True)
        return True

    # ---- the whole run ----

    def run(self, limit: int | None = None) -> None:
        if not limit:
            self.log("no --limit given — nothing to do. Every job is generated "
                     "fresh (no prompts.txt involved), so run --limit N for N videos.")
            return

        self.log(f"=== start: generating {limit} job(s) end to end ===")

        completed = 0
        for i in range(1, limit + 1):
            reason = self.blocked()
            if reason:
                self.log(f"stopping: {reason}")
                break
            try:
                if not self.run_job(i, limit):
                    break
                completed += 1
            except Exception as e:
                self.log(f"    FAILED: {e}")
                (self.out_root / "errors.log").open("a", encoding="utf-8").write(
                    f"\n=== {datetime.now()} :: job {i}/{limit}\n{traceback.format_exc()}")
                self.ledger.record(f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}",
                                   status="failed", error=str(e)[:400])
                # This is a clean, logged failure (we're inside the except
                # block, not dead) - the job reached a real resolution, it
                # just wasn't a good one. Leave the pipeline's existing
                # "log it and move on to the next prompt" behavior alone;
                # the marker is only meant to flag a prompt that never got
                # ANY resolution because the whole process died.
                self._inflight_marker.unlink(missing_ok=True)
                self.comfy.free_memory()
                time.sleep(10)
            finally:
                self.clear_staging()

            cooldown_n = self.run_cfg.get("cooldown_every_n_jobs")
            if cooldown_n and i % cooldown_n == 0 and i < limit and not self.blocked():
                if not self.periodic_cooldown():
                    self.log("stopping: ComfyUI did not come back after scheduled cooldown")
                    break

        self.comfy.free_memory()
        self.log(f"=== finished: {completed}/{limit} completed — {self.ledger.summary()} ===")

    def clear_staging(self) -> None:
        """Sweep _lucid_staging clean after every job, success or failure.

        Belt and suspenders on top of comfy_client's own per-file cleanup.
        Different custom node packs report their save path back to
        ComfyUI's API in subtly different shapes — confirmed empirically:
        Pixaroma's nodes report a clean relative subfolder, but Video Helper
        Suite's VHS_VideoCombine reports its own absolute path string
        instead (traced to core ComfyUI's get_save_image_path — it returns
        os.path.dirname of an absolute filename_prefix as-is), and that
        left real files behind in _lucid_staging/upscale and
        /interpolate even though the path math should still resolve
        correctly through it. Rather than chase every custom node's own
        path-reporting quirk, wipe the whole staging root after each job:
        by this point everything in it is already a duplicate of something
        this pipeline safely copied into its own output tree.
        """
        staging = Path(self.cfg["comfy"]["output_dir"]) / "_lucid_staging"
        if not staging.exists():
            return
        for f in staging.rglob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass

    # ---- helpers ----

    def status(self) -> dict:
        return {
            "comfyui_running": self.comfy.is_up(),
            "vram": self.comfy.vram() if self.comfy.is_up() else {},
            "gpu_temp_c": gpu_temp(),
            "queue_depth": self.comfy.queue_depth() if self.comfy.is_up() else None,
            "ledger": self.ledger.summary(),
            "free_disk_gb": round(shutil.disk_usage(self.out_root).free / 1e9, 1),
            "stop_file_present": STOP_FILE.exists(),
            "final_folder": str(self.out_root / self.cfg["paths"]["final_folder"]),
        }

    def check(self) -> list[str]:
        """Validate everything before a long unattended run.

        Loads each workflow, confirms every title and setting named in
        config.yaml actually exists. Five seconds now saves a wasted night.
        """
        problems = []
        if not self.comfy.is_up():
            problems.append("ComfyUI is not responding on " + self.comfy.base)
        if self.comfy.input_dir and not self.comfy.input_dir.exists():
            problems.append(f"comfy.input_dir does not exist: {self.comfy.input_dir}")
        prompts_file = ROOT / self.cfg["paths"]["prompts_file"]
        if not prompts_file.exists():
            problems.append(f"prompt file missing: {prompts_file}")

        for stage in self.stages:
            wf = self.wf_dir / stage["workflow"]
            if not wf.exists():
                problems.append(f"[{stage['name']}] workflow file missing: {wf}")
                continue
            try:
                graph = ComfyClient.load_workflow(wf)
            except ComfyError as e:
                problems.append(f"[{stage['name']}] {e}")
                continue
            for item in stage.get("set", []):
                try:
                    node = graph[ComfyClient.find_by_title(graph, item["title"])]
                    if item["from"] == "seed":
                        if not ({"seed", "noise_seed"} & set(node["inputs"])):
                            problems.append(f"[{stage['name']}] '{item['title']}' has no seed setting")
                    elif item["key"] not in node["inputs"]:
                        problems.append(
                            f"[{stage['name']}] '{item['title']}' ({node['class_type']}) has no "
                            f"'{item['key']}'. It has: {sorted(node['inputs'])}")
                except ComfyError as e:
                    problems.append(f"[{stage['name']}] {e}".replace("\n", " "))
        return problems


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Lucidscapes pipeline")
    ap.add_argument("command", choices=["run", "status", "check", "inspect"])
    ap.add_argument("workflow", nargs="?", help="for 'inspect': a workflow filename")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pipe = Pipeline()

    if args.command == "status":
        print(json.dumps(pipe.status(), indent=2))
    elif args.command == "inspect":
        print(ComfyClient.describe(ComfyClient.load_workflow(pipe.wf_dir / args.workflow)))
    elif args.command == "check":
        problems = pipe.check()
        if problems:
            print("Problems found:\n")
            for p in problems:
                print("  x " + p)
            return 1
        print("All good. Workflows, titles and settings all line up.")
    elif args.command == "run":
        if not pipe.comfy.is_up():
            print("ComfyUI is not running. Start it first.", file=sys.stderr)
            return 2
        pipe.run(limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
