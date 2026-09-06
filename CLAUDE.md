# Lucidscapes — project context for Claude Code

AI-generated ambient/cozy cinematic content for YouTube, Instagram, TikTok.
This folder is the unattended ComfyUI generation pipeline. Local RTX 5090.

**Fully autonomous by design.** There is no human-authored prompt anywhere in
the automated flow. QwenVL invents the still-image scene from scratch, Flux
renders it, QwenVL looks at that render and writes the motion instruction,
WAN animates it, then upscale and interpolate finish it. `run --limit N`
means "generate N brand-new videos," not "process N lines from a file."
`prompts/prompts.txt` and `read_prompts()` still exist only for the
interactive MCP path (`mcp/lucid_mcp.py`'s `add_prompts` tool) — the
automated `run` command never reads that file.

## Architecture
- `runner/pipeline.py` — the runner. Loops `run_job()` `--limit` times (each
  call is one fully independent, autonomously-generated video), pushing each
  job through every stage in `config.yaml`'s `stages:` list in order, one at
  a time, models unloaded between every stage (`free_vram_between_stages:
  true` in config — never turn this off, several large models share one
  GPU). stdout/stderr are reconfigured to UTF-8 at import time (see gotchas
  below — Windows' default console codepage will otherwise crash on an
  emoji node title mid-run, not just at printing time).
- `runner/comfy_client.py` — thin wrapper around ComfyUI's HTTP + websocket
  API. Handles workflow submission, live progress logging, OOM retry,
  cleanup of ComfyUI's own duplicate output copies.
- `mcp/lucid_mcp.py` — optional MCP server for daytime interactive control.
  NOT involved in autonomous runs.
- All generation settings (resolution, steps, CFG, sampler, RIFE/SeedVR2
  values) live inside the exported ComfyUI workflow JSONs in `workflows/`,
  never in config.yaml. To change one: change it in ComfyUI, re-export with
  Workflow -> Export (API), overwrite the file. The one deliberate exception
  is output save-paths (see "Output folder consolidation" below) — those are
  overridden at runtime from config.yaml on purpose, so they survive a
  re-export instead of reverting to someone's personal ComfyUI folder names.

## Current pipeline stages (6, in this order)
1. `prompt` — QwenVL (`00_prompt_gen.json`, node title `PROMPT_GEN`) invents
   an original Flux still-image prompt from scratch. No idea/topic input —
   see "Forced scene variation" below for how it's kept from just describing
   the same cabin-in-snow every time. `produces: prompt` / `sets:
   image_prompt`.
2. `still` — Flux 2 Dev (`01_flux_still.json`). Turbo LoRA is **disabled**
   (the "Enable Turbo LoRA" boolean primitive, node `98:104`, is `false`) —
   20 steps, not 8. Toggling that one boolean also switches the steps
   primitive via `Switch(steps)`, so it's a one-field change if this ever
   needs reverting.
3. `video_prompt` — QwenVL again (`02b_video_prompt_gen.json`), looking at
   the finished still image (`LOAD_IMAGE`) and writing the WAN motion
   instruction based on what's actually visible. `produces: prompt` /
   `sets: video_prompt`. Its output also drives `camera_pose` — see below.
4. `video` — **WAN 2.2 Fun Camera Control** (`02_wan2.2_i2v.json`),
   replacing MiniMax (`02_minimax_i2v.json`, now unused but left in
   `workflows/` for reference). 5-second clips. See "WAN camera control"
   below for how `camera_pose` gets set from QwenVL's text.
5. `upscale` — SeedVR2 (`03_seedvr2.json`). Export frame rate locked to 24
   (was found set to 8 — a leftover mistake, not intentional; fixed).
6. `interpolate` — RIFE (`04_rife.json`), locked settings: rife49,
   multiplier 2, ensemble on, fast_mode off, `scale_factor: 1.0` (must be a
   float in the exported JSON — `1` as a bare int fails this project's
   `validate_workflow` enum check, though ComfyUI itself tolerates it fine).

## produces: / sets: / from: template — IMPLEMENTED, not just designed
`run_job()`'s per-stage loop in `pipeline.py` fully implements this now:
- `from: template` — `item["template"].format(**ctx)`; the template string
  names ctx keys directly in `{curly_braces}` (`{image_prompt}`,
  `{scene_brief}`, etc.), no separate vars: mapping.
- A stage with `produces: prompt` does NOT let its output become
  `ctx["previous_file"]`. Its `.txt` file is read
  (`encoding="utf-8", errors="replace"`) and the text overwrites
  `ctx[stage["sets"]]` instead — `previous_file` keeps pointing at the last
  real media file so the next stage's `LOAD_IMAGE`/`LOAD_VIDEO` still gets
  a picture, not a text file. Don't remove that guard.
- Locating that stage's own output file is NOT done by matching ComfyUI's
  reported job outputs (`CR Save Text To File` reports none at all — see
  gotchas). `run_stage()` instead pre-creates the destination directory
  before submitting the graph, and afterward reads
  `self.out_root / stage["save_to"] / f"{job_name}{stage['extensions'][0]}"`
  directly — a path it can predict exactly, since `file_name` is always set
  to `job_name` via `set:` and `output_file_path` is baked into the workflow
  JSON as that same folder.
- `comfy_client.wait()` had to change too: it used to require
  `entry.get("outputs")` to be non-empty before considering a job done, but
  `CR Save Text To File` never populates that field even on a clean run —
  confirmed live, `status.completed: true` with `outputs: {}`. `wait()` now
  also accepts `status.get("completed")` or `status_str == "success"`.
  Without this fix a `produces: prompt` stage hangs until its timeout even
  though ComfyUI finished successfully seconds in.

## Forced scene variation (Python decides the subject, not QwenVL alone)
A small local QwenVL (Qwen3-VL-2B-Instruct), given the same static
instruction every job, collapses onto whatever sits closest to its few-shot
example — confirmed repeatedly: multiple jobs in a row all came back "small
stone cabin nestled in snowy ridgeline/hillside/cliffside," and even after
adding an explicit "vary the subject" instruction, roughly 3 in 10 jobs in
one full batch still came back as a **verbatim copy of the example text**,
completely ignoring a differently-themed brief. Asking the model to invent
its own variety isn't reliable; forcing it is.

- `build_scene_brief()` (module-level in `pipeline.py`) randomly assembles a
  concrete brief every job: one `SCENE_SUBJECTS` entry (a `(phrase,
  keyword)` pair — e.g. `("a stopped vintage train", "train")`), one
  `SCENE_WEATHER`, one `SCENE_TERRAIN`, and two `SCENE_ACCENTS`. The
  `prompt` stage's instruction requires QwenVL to build the scene around
  exactly that brief, with an explicit "this is a style reference only,
  copying it is WRONG" warning ahead of the one shown example.
- `brief_adhered(text, subject_keyword)` — a plain substring check — runs
  right after the `prompt` stage produces its text. If the subject keyword
  never appears, `run_job()` bumps `ctx["seed"]` (a same-seed retry would
  likely reproduce the same fallback, since `torch.manual_seed(seed)` makes
  QwenVL's sampling deterministic-ish) and re-runs just that one stage,
  once. It never hard-fails the job — worst case, a still-non-adherent
  result after the retry is accepted and logged.
- Keep `SCENE_SUBJECTS`' keyword half honest: it must be a word QwenVL would
  plausibly write when genuinely describing that subject, or the adherence
  check will false-negative and burn a retry for no reason.
- Validated with a small sample (2 jobs, both passed on the first attempt,
  no retry needed) — worth watching over a larger batch before assuming
  it's fully solved.

## WAN camera control (camera_pose is a separate structured field, not text)
WAN's `WanCameraImageToVideo` node does NOT expose a `camera_pose` widget
itself — it takes an optional `camera_conditions` link of type
`WAN_CAMERA_EMBEDDING`, produced by a **separate** node,
`WanCameraEmbedding`, whose `camera_pose` COMBO has exactly 9 choices:
`Static, Pan Up, Pan Down, Pan Left, Pan Right, Zoom In, Zoom Out, Anti
Clockwise (ACW), ClockWise (CW)`. That embedding node was **not present** in
the workflow file as originally exported (camera direction was being driven
purely by text, e.g. "Camera pan down, fire glowing...") — it was added to
`02_wan2.2_i2v.json` as node id `300`, titled `CAMERA_POSE`, wired
`width`/`height`/`length` from the same sources `WanCameraImageToVideo`
already used (so they can't drift out of sync), and its `camera_embedding`
output feeds `WanCameraImageToVideo`'s `camera_conditions` input.

Since QwenVL only ever produces text, `derive_camera_pose()` in
`pipeline.py` recovers WAN's exact enum string with a plain keyword match —
reliable specifically because the `video_prompt` stage's own instruction is
constrained to always open with exactly one of six fixed phrases ("Slow pan
left/right/up/down", "Slow zoom in/out toward..."), a vocabulary chosen to
line up 1:1 with 6 of WAN's 9 choices (Static and the two rotation options
excluded — they don't fit the slow-cinematic brand). No second QwenVL call
needed just to classify what the first call already committed to.
`ctx["camera_pose"]` gets set immediately after `ctx["video_prompt"]` in
`run_job()`, and defaults to `"Static"` if nothing matches.

Other titles in `02_wan2.2_i2v.json`, renamed for config.yaml consistency:
`Load Image` -> `LOAD_IMAGE`, `CLIP Text Encode (Positive Prompt)` ->
`VIDEO_PROMPT`, and one of two identically-titled `KSampler (Advanced)`
nodes (the one that actually takes `noise_seed`, `add_noise: "enable"`) ->
`SEED_KSAMPLER` — the other stays untitled-generic since nothing targets it.

**WAN produces no audio.** `03_seedvr2.json` and `04_rife.json`'s
`VHS_LoadVideo` nodes used to pipe an `audio` output into their
`VHS_VideoCombine` nodes (left over from MiniMax, which had a real
TTS/audio branch). `VHS_LoadVideo`'s audio output is a `LazyAudioMap` —
ffmpeg extraction only actually runs when something downstream reads it —
so wiring it to a video with no audio track crashes with "VHS failed to
extract audio" the moment `VHS_VideoCombine` touches it. Both `audio` links
were removed; there is currently no audio anywhere in this pipeline.

## Output folder consolidation (`_lucid_staging`)
Every exported workflow's own save node originally pointed at the author's
personal ComfyUI folder structure (`Images/Flux/text_to_image/Cozy
cinematic`, `Videos/minimax/Cozy Cinematic`,
`Upscaled_FrameInterpolated/Cozy Cinematic/{upscale,interpolate}` — four
different, scattered, absolute paths). All four media stages now override
their save node's output path via `set:` in config.yaml (survives every
future re-export) to one shared root:
`C:/ComfyUI/output/_lucid_staging/<stage>/`. `comfy_client._cleanup_source`
still deletes the ComfyUI-side original right after each file is downloaded
into this project's own organized `output/` tree, and now retries briefly
and prints a visible warning if it can't (it used to fail completely
silently). On top of that, `Pipeline.clear_staging()` unconditionally wipes
the whole `_lucid_staging/` tree after every job (success or failure) —
belt and suspenders, because different custom node packs report their save
path back to ComfyUI's API in subtly different shapes (Pixaroma reports a
clean relative subfolder; `VHS_VideoCombine`, via core ComfyUI's
`get_save_image_path`, reports the `os.path.dirname` of an absolute
`filename_prefix` as-is — an absolute string, not relative — and that
still left real files behind in `_lucid_staging/upscale` and
`/interpolate` even though the path math should have resolved correctly
through it). Never assume the per-file cleanup alone is enough; the sweep
is what actually guarantees nothing accumulates.

## Known custom-node gotchas already solved — don't rediscover these
- **Pixaroma custom nodes** (Seed Pixaroma, PixaromaPrompt, PixaromaSaveImage,
  etc.) often store their real value inside a JSON-encoded STRING field
  (e.g. SaveImageState = '{"folder":"...","pattern":"..."}'), not a plain
  widget. A native Python dict prints with single quotes and a space
  ({'key': 1}); JSON-as-a-string prints with double quotes and no space
  ({"key":1}) — that's how you tell them apart from `inspect` output. Use
  `json_field:` in config to target the real field inside the blob; the
  code preserves whatever original type (string vs dict) it finds, since
  sending the wrong type can silently fail.
- **Seed node**: switched away from Seed Pixaroma entirely to ComfyUI's
  native RandomNoise (title "RandomNoise", plain noise_seed field) for the
  `still` stage — this is the CONFIRMED WORKING setup, verified by
  comparing two renders' actual pixel content, not just their recorded seed
  values. Do not reintroduce Seed Pixaroma.
- **Node titles, not IDs**: always target nodes by `_meta.title` in config,
  never numeric node id — ids renumber on any graph edit. Watch for
  duplicate titles within one workflow (WAN's two `KSampler (Advanced)`
  nodes both had the same title until one was renamed) — `find_by_title`
  silently returns whichever node it encounters first in dict-iteration
  order, no error.
- **UI-format vs API-format JSON**: workflows/*.json must be exported via
  Workflow -> Export (API). The normal Save button produces a different
  format (`nodes`/`links` keys) that `load_workflow()` explicitly rejects
  with a clear message.
- **`CR Save Text To File` never creates its own output directory** — plain
  `open()`, no `os.makedirs`. `run_stage()` pre-creates
  `self.out_root / stage["save_to"]` before submitting a `produces: prompt`
  stage's graph for exactly this reason.
- **`CR Save Text To File` writes with no explicit encoding** — Python's
  locale default (cp1252 on this Windows box), which mangles any non-ASCII
  character QwenVL produces (a stray em-dash showed up as `\ufffd`). Not
  fixable from this project's side (third-party node source); worked around
  by instructing QwenVL, in every prompt-gen instruction, to use only plain
  ASCII punctuation.
- **Emoji node titles crash on Windows' default console codepage** — Video
  Helper Suite's own `Video Combine 🎥🅥🅗🅢` title, printed during normal
  progress logging, raises `UnicodeEncodeError` under cp1252 with no
  `PYTHONIOENCODING` override. Fixed once, generally, by reconfiguring
  `sys.stdout`/`sys.stderr` to UTF-8 (`errors="replace"`) at the top of
  `pipeline.py` — don't reach for `PYTHONIOENCODING` as the fix, it doesn't
  help anyone running the script without setting it first.
- **`config.yaml`/`ledger.json` need explicit `encoding="utf-8"` on every
  read/write** — Windows' `Path.read_text()`/`write_text()` default to the
  locale encoding otherwise, silently corrupting anything non-ASCII (this
  is what broke the em-dash-containing emoji title before the fix above was
  even relevant).
- **RIFE VFI does not reliably honor `/interrupt` mid-computation** — when
  stopping a run mid-flight, expect the current RIFE interpolation to
  finish on its own rather than abort immediately; nothing further gets
  queued behind it once the driving Python process is killed.
- **`validate_workflow`'s enum matching is stricter than ComfyUI itself** —
  it flagged `scale_factor: 1` (int) against a float-typed enum
  (`[0.25, 0.5, 1.0, 2.0, 4.0]`) as invalid even though real runs had
  already used that exact value successfully many times. Harmless to tidy
  (`1` -> `1.0`) but not a sign anything is actually broken if you see it
  elsewhere.

## VRAM / stability
- Single RTX 5090, ~34GB. Six models now share it across stages (QwenVL x2,
  Flux, WAN, SeedVR2, RIFE) — never load two heavy models at once; the
  unload-between-stages design is load-bearing, not optional.
- QwenVL nodes (`AILab_QwenVL`) must keep `keep_model_loaded: false`, or
  `/free` can't evict them between stages regardless of
  `free_vram_between_stages`.
- `max_gpu_temp_c: 84` pauses and cools rather than pushing through.
- OOM handling: interrupt, free_memory(), wait, retry same stage
  (`oom_retries`) — this is a resumable retry, not a crash.

## Ledger / job identity
Every job invents its own prompts, so there is no fixed prompt text to hash
and no "have I already done this one?" question to ask — `run --limit N`
always means "generate N more," full stop. The ledger (`output/ledger.json`)
is a history/stats log now, not a dedup gate: `Ledger.record()` takes a raw
key directly (no more `stable_key()`/content-hash step — that function and
`Ledger.key()`/`Ledger.done()` were removed along with the old
prompts.txt-driven dedup flow). `job_name` itself
(`{timestamp}_{uuid4 hex[:8]}`) is already a unique per-job identifier and
is used as the ledger key as-is.

## Progress logging
Live per-node status comes from a websocket listener in comfy_client.py,
routed through plain timestamped log lines (not carriage-return overwriting
— that broke across this project's actual terminal/logging setup). Samplers
(KSampler, SamplerCustomAdvanced, KSamplerAdvanced) report real step
counts; single-pass nodes (VAEDecode, save nodes, etc.) never do, in any
workflow — that's normal, not a bug, and the code pings elapsed time for
those instead of looking frozen.

## Working preferences
- No automated quality control anywhere in this pipeline, by design. It
  renders and saves; the user reviews output/FINAL/ manually. Do not add
  scoring, filtering, or auto-rejection unless explicitly asked. (The
  brief-adherence retry in "Forced scene variation" above is not an
  exception to this — it's a mechanical text-matching safety net for a
  specific known failure mode, not aesthetic/quality judgment.)
- YouTube titles lead with curiosity/emotion, not scene description.
- Snow/ice/cold environments are the highest-performing content territory.
- Real prompt style reference (both still and motion) came from the user's
  own past "Cozy cinematic niche" / "Cozy cinematic niche 2" chats — terse
  comma-separated concrete fragments, NOT flowing/poetic prose, with a
  near-fixed technical closing block ("Arri Alexa 35mm anamorphic, Kodak
  Vision3 500T, cold desaturated environment, only [warm sources] warm,
  zero warm ambient light on [cold elements], deep shadow detail, visible
  film grain, no people, no golden hour, no HDR"). Motion prompts follow an
  equally fixed pattern: one named camera movement stated once at the
  start, every other element in the frame explicitly tagged "completely
  still"/"completely steady", only genuinely ambient motion (falling
  snow/rain, drifting mist, rising steam) allowed to move, ending in a
  short two-to-four-word mood tag. Both QwenVL instructions are built
  around real examples in this style, not invented from scratch — see
  `00_prompt_gen.json`/`02b_video_prompt_gen.json` and the matching
  `config.yaml` templates for the exact wording.

## Status as of this session
- The full 6-stage autonomous chain (QwenVL -> Flux -> QwenVL -> WAN ->
  SeedVR2 -> RIFE) has completed end to end multiple times, including one
  full batch of 10/10 successful jobs (~11 min/job).
- MiniMax was fully replaced by WAN 2.2 Fun Camera Control this session;
  `02_minimax_i2v.json` is no longer referenced anywhere and could be
  deleted if it's confirmed nobody wants it as a reference.
- A 3-job follow-up batch (regenerating replacements for 3 jobs that hit
  the pre-fix variation bug) was interrupted by the user partway through:
  job 1 completed clean, job 2 was killed mid-RIFE-interpolate (no partial
  FINAL file produced), job 3 never started. Nothing is currently running.

## OPEN TASK: prompt quality is worse than the reference chats — diagnose before changing the model
User's report: QwenVL's prompts (both the Flux still prompt and the WAN
motion prompt) aren't as good as the ones written in the user's own past
"Cozy cinematic niche" / "Cozy cinematic niche 2" chats (see "Working
preferences" above for that style spec). User asked whether this is because
the model is too small (Qwen3-VL-2B-Instruct) and whether to size up.

**This is actually two separable problems — don't conflate them, and don't
size up the model before checking which one is still live:**

1. **Repetition / copying the few-shot example.** Already documented above
   under "Forced scene variation" — confirmed multiple times pre-fix, only
   2 post-fix data points so far, both clean. This is a *methodology*
   problem (unconstrained "invent your own scene" naturally collapses onto
   whatever example it was shown), not primarily a raw-capability problem —
   `build_scene_brief()` already exists specifically to force variety rather
   than hope for it. **Do not treat this as solved.** It has not been
   checked at any real scale yet.

2. **Prose/style fidelity vs. the reference chats** — terse comma-separated
   concrete fragments (not flowing prose), hitting every required technical
   closing-block element every single time, evocative rather than generic
   word choice. This kind of precise adherence to a fairly detailed style
   spec is something small models are genuinely, measurably worse at than
   large ones — this is where model size can legitimately be the answer,
   unlike problem 1.

**Do this in order, don't skip to a model swap:**

1. Run a real batch — 10 to 20 jobs, not 2 — with the CURRENT
   Qwen3-VL-2B-Instruct setup exactly as it stands (forced scene variation
   already active, don't touch it yet).
2. Read every generated `image_prompt` and `video_prompt` text
   (`output/00_prompts/*.txt`, `output/02b_video_prompts/*.txt`) across that
   whole batch. Check separately for:
   - **Variety**: does the subject actually differ job to job now, or is the
     pre-fix collapse/copying still showing up despite `build_scene_brief()`
     and the `brief_adhered()` retry?
   - **Style fidelity**: terse comma-fragment structure held throughout, or
     drifting into flowing prose? Every required closing-block element
     present every time? Word choice evocative and specific, or generic?
3. Report both findings to the user as two separate verdicts before doing
   anything else. If variety is still broken, that's a `build_scene_brief()`/
   `brief_adhered()` bug to fix — not a reason to size up the model. If
   variety is solid but style fidelity is still weak, THAT is the actual
   case for sizing up.
4. If and only if step 3 confirms a real style-fidelity gap: change
   `model_name` to `"Qwen3-VL-8B-Instruct"` in both `00_prompt_gen.json` and
   `02b_video_prompt_gen.json` (the `AILab_QwenVL` node's dropdown already
   lists it — no new install). Keep `quantization` reasonable for VRAM
   headroom (`"8-bit (Balanced)"` is a sound starting point) and
   `keep_model_loaded: false` unchanged — that setting is load-bearing for
   VRAM discipline, see "VRAM / stability" above, do not touch it. Re-export
   both workflows, run `python -m runner.pipeline check`, then run a
   same-sized fresh batch for a fair before/after comparison — not a smaller
   one, or the comparison won't mean anything.
5. Do NOT build any new automated scoring or style-adherence checking as
   part of this — that would contradict "Working preferences" above
   (no automated quality control beyond the existing narrow subject-keyword
   safety net). The comparison in steps 2 and 4 is a manual read-through,
   same as every other quality decision in this project.
6. Skipping straight to 8B without doing steps 1-3 first risks spending
   VRAM/load-time budget on a fix for the wrong problem, and leaves whether
   `build_scene_brief()` actually works at scale permanently unverified.
