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
import random
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
# The Fun-Camera-Control WAN workflow (WanCameraEmbedding's structured
# camera_pose COMBO) has been dropped in favor of a plain WanImageToVideo
# workflow with no structured camera field at all — camera direction is now
# driven purely by QwenVL's text, same as the rest of the motion
# description. Letting QwenVL pick freely from six named directions was
# tried first and failed the same way the scene subject did: confirmed in
# practice, 9 of 10 jobs in one batch came back "Slow pan left" regardless
# of which direction the instruction listed first or how the image looked.
# So Python forces the direction per job (build_scene_brief() below is the
# same pattern for the scene subject) and the video_prompt stage's
# instruction requires QwenVL to open with that exact phrase rather than
# choosing one itself.
CAMERA_MOVES = [
    ("Slow pan left", "pan left"),
    ("Slow pan right", "pan right"),
    ("Slow pan up", "pan up"),
    ("Slow pan down", "pan down"),
    ("Slow zoom in toward the subject", "zoom in"),
    ("Slow zoom out from the subject", "zoom out"),
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


def brief_adhered(text: str, subject_keyword: str) -> bool:
    """False almost always means the model fell back to reciting its
    few-shot example instead of building a scene around the given brief —
    confirmed in practice: 3 of 10 jobs in one batch came back with the
    example's exact wording ("small white wooden house perched on steep
    mountain ridge...") despite each having a completely different subject
    in their brief (a train, a lighthouse, a train). A same-seed retry
    would likely reproduce the same failure, so run_job() must also bump
    the seed when this returns False."""
    return subject_keyword.lower() in text.lower()


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
        self.ledger = Ledger(ROOT / self.cfg["paths"]["ledger"])
        self.deadline = datetime.now() + timedelta(hours=self.run_cfg["max_hours"])
        self.out_root.mkdir(parents=True, exist_ok=True)

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
        subprocess.Popen(cmd, shell=True)
        for _ in range(60):
            time.sleep(5)
            if self.comfy.is_up():
                self.log("ComfyUI is back")
                return True
        self.log("ComfyUI did not come back")
        return False

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

        dest = out_dir / f"{ctx['job_name']}{Path(matches[-1]['filename']).suffix}"
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
               "scene_brief": scene_brief, "camera_move": camera_move,
               "previous_file": None}
        # Which ctx key a produces:prompt stage sets, and the keyword its
        # text must contain to count as actually having followed the
        # forced brief/camera-move rather than drifting off it.
        required_keywords = {"image_prompt": scene_subject_keyword,
                             "video_prompt": camera_move_keyword}

        self.log(f"--- job {index}/{total}: {job_name}  "
                 f"brief: {scene_brief}  camera: {camera_move}")
        started = time.time()

        for stage in self.stages:
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
                            # QwenVL ignored the forced brief/camera-move and
                            # fell back to its few-shot example instead — a
                            # same-seed retry would likely repeat it, so
                            # bump the seed (also reused by every later
                            # stage, which is fine — it's a per-job value)
                            # and try this one stage again, once.
                            self.log(f"    {stage['name']:<12} ignored the required phrasing "
                                     f"(no '{required}' in the output) "
                                     f"— retrying with a fresh seed")
                            ctx["seed"] = random.randint(1, 2**31 - 1)
                            result = self.run_stage(stage, ctx)
                            text = Path(result).read_text(encoding="utf-8", errors="replace").strip()
                            if not brief_adhered(text, required):
                                self.log(f"    {stage['name']:<12} still ignored it "
                                         f"after retry — continuing anyway")
                        ctx[stage["sets"]] = text
                        self.log(f"    {stage['name']:<12} ok in {time.time() - t0:>5.0f}s  "
                                 f"-> {stage['sets']}: {text[:70]}...")
                    else:
                        ctx["previous_file"] = result
                        self.log(f"    {stage['name']:<12} ok in {time.time() - t0:>5.0f}s  "
                                 f"-> {Path(result).name}")
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
                self.comfy.free_memory()
                time.sleep(10)
            finally:
                self.clear_staging()

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
