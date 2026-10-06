# Lucidscapes — project context for Claude Code

AI-generated ambient/cozy cinematic content (snow/rain/cold scenes with only
practical warm light) for Facebook, Instagram (and formerly YouTube).
This folder is the unattended ComfyUI generation pipeline. Local RTX 5090.

Last rewritten 2026-10-06 from the actual code/config/Output state (the
previous version described a WAN/QwenVL-invents-everything era that no
longer exists). If something here disagrees with `config.yaml` or
`runner/pipeline.py`, the code wins — fix this file.

## How it runs now
- **Prompts are NOT invented any more.** The `prompt` stage has
  `source: prompts_file` in config.yaml: one still-image prompt per job is
  taken in order from `prompts/lucidscapes_prompts_batch2.json` (dict with
  `style_lock` + a `prompts` list of 100 `{id, category, prompt}`), position
  persisted in `<output_root>/prompts_position.json`. When the file is
  exhausted, `run_job()` returns False and the run stops. The old QwenVL
  invention path (`build_scene_brief`, `SCENE_*`, `brief_adhered`) is still in
  `pipeline.py` and config (workflow `00_prompt_gen.json`) — deleting the
  `source:` line reverts to it. `prompts/prompts.txt` is gone.
- **Start/stop**: `resume_production.bat` -> `resume_production.py`. It
  rolls the prompt position back if a job was cut off mid-flight (see
  inflight marker below), starts ComfyUI if down (cwd must be `C:/ComfyUI`),
  tails ComfyUI's log, runs `python -m runner.pipeline run --limit 500`,
  and stops ComfyUI on exit (Ctrl+C = clean stop, GPU freed). `STOP` file in
  the repo root also halts a run.
- **Output root is `output/` inside this repo folder** (`Code/output`)
  (config `paths.output_root: output`, gitignored; it was `G:/LucidScapes Pipeline Output Folder`
  until 2026-10-06, when G: was unmounted; merged into the repo folder that day). Ledger, approvals, state JSONs
  and all media live there. Guard: `pipeline.py`, `resume_production.py` and
  `check_approvals.py` refuse to run if `ledger.json` / `pending_approvals.json`
  is missing there (set `LUCID_ALLOW_NEW_ROOT=1` only for a deliberate fresh
  start). `check_approvals.py` logs such failures to
  `Code/check_approvals_errors.log`. The scheduled task "Lucidscapes Check
  Approvals" runs `pythonw -m runner.check_approvals` every 10 min with start-in
  `E:\AI\Automated Pipelines. Lucidscapes\Code` - if the project moves
  again, update that task's working directory too (it silently failed for a
  week with 0x8007010B after the move from `C:\ComfyUI\output\Work\...`).

## Pipeline stages (7, in order — `config.yaml` `stages:`)
1. `prompt` — pulls next Flux prompt from the prompts JSON (see above).
2. `still` — Flux 2 Dev (`01_flux_still.json`), saved via Pixaroma node;
   `01_flux_still.json.bak_pre_lora05` is the pre-LoRA-0.5 backup.
3. `video_prompt` — QwenVL (`02b_video_prompt_gen.json`) looks at the still
   and writes a **four-sentence LTX-style motion paragraph** that must open
   with a Python-forced camera move (`CAMERA_MOVES`: pan left/right/up/down,
   push-in, pull-back) **plus** `TITLE:/DESCRIPTION:/TAGS:` upload metadata in
   the same response. `split_motion_and_metadata()`/`parse_metadata()` split
   them. `brief_adhered()` retries once with a new seed if the camera keyword
   or `title:` is missing.
4. `video` — **LTX 2.5 i2v** (`02_ltx2.5_i2v.json`; chosen over MiniMax and
   WAN as measurably better). Two-pass: half-res base, 2x spatial upsample,
   light refine. After the render, `Pipeline.grade_video()` runs an ffmpeg
   tonal grade (+ unsharp 0.15, env `LUCID_SHARPEN`) in place. Its own audio
   is unused.
5. `audio` — Stable Audio 3 (`02b_stable_audio.json`) listens/looks at the
   LTX clip, generates 20s mp3, `shape_audio()` (ffmpeg) keeps the first 10s
   with a soft first second. Output goes to `ctx["audio_file"]` via
   `sets_file:` so `previous_file` stays the video.
6. `upscale` — SeedVR2 (`03_seedvr2.json`), tuned for 2K (resolution 1440,
   blocks_to_swap 0, batch_size 8). Works on the native 5s/121 frames
   (241 frames measured ~1100s vs ~450s). `.bak_pre2k`,
   `.bak_before_batch25`, `workflows/test2k/` are tuning leftovers.
7. `interpolate` — RIFE (`04_rife.json`) doubles frames to make ~10s and
   attaches the audio via a `LOAD_AUDIO` node.
Then the file is copied to `FINAL/` and the ledger entry written
(`status: done`), then social staging runs (below). Typical job ~15 min
(median in ledger), up to ~22 min.

`02_minimax_i2v.json`, `minimax_r2v.json` are unused reference. The WAN
workflows were deleted (git shows them removed; they are in the initial
commit).

All generation settings live in the exported workflow JSONs, never in
config.yaml. Change in ComfyUI, re-export with Workflow -> Export (API),
overwrite. Exception: save paths — overridden from config via `set:` to
`C:/ComfyUI/output/_lucid_staging/<stage>/` and swept by
`Pipeline.clear_staging()` after every job.

## Job bookkeeping
- `ledger.json` — history log keyed by `job_name` (`{timestamp}_{uuid8}`).
  187 entries as of 2026-09-29: 157 done, 30 failed.
- `generated_videos_data/generation_data.json` — the one persistent record
  per `generation_id` (= job_name): image prompt, motion prompt, title,
  description, tags. The scratch `.txt` files in `00_prompts/` /
  `02b_video_prompts/` are deleted right after reading.
- `inflight_prompt.json` — written when a prompt is consumed, removed at job
  end (success or logged failure). If present at startup, the process died
  mid-job and `resume_production.py` rolls `prompts_position.json` back so
  that prompt is regenerated. Clean logged failures are NOT retried.
  `*.batch1_backup.json` are old copies (position 52).
- `errors.log` (tracebacks), `run.log` (everything, ~2.4MB).
- Periodic cooldown: every `cooldown_every_n_jobs: 5` ComfyUI is fully
  stopped and restarted (`cooldown_minutes: 5`; the comment in config still
  says 15 — stale). `max_gpu_temp_c: 84`, `min_free_disk_gb: 40`.
- Seeds: per-job random seed, used for base+refine noise and all stages.

## Social upload — WIRED into the job loop, live
`run_job()` calls `stage_social_upload()` (`runner/social/approvals.py`)
after each finished video; failures there are logged and never fail the job.
- **YouTube is PAUSED/removed from staging — the channel was banned**
  (working theory: automated-posting behavior). `youtube.py` and its queue
  still exist but nothing enqueues. Re-adding = add to `platforms` and
  re-enable `youtube.enqueue()` in `stage_job`.
- Staging now: create a **Facebook unpublished draft** immediately, post a
  Discord approval message (title, scene text, FB link), record in
  `pending_approvals.json` with `platforms: [facebook, instagram]`.
- **Instagram is NOT staged at generation time** — a resumable-upload
  container expires (error 2207020 "Media expired") before a human reacts.
  `check_approvals.py` creates the container and publishes back-to-back at
  approval time (`video_path` + `ig_caption` are stashed in the entry).
- `runner/check_approvals.py` (run every 10-15 min via Scheduled Task): reads
  Discord ✅/❌ (bot's own pre-react excluded), publishes to each staged
  platform not yet published, marks rejected on ❌. Per-platform success flags
  (`published_facebook`, `published_instagram`) are set immediately after each
  call so partial failures only retry the failed platform.
- `runner/social/rate_limit.py` throttles publishing (not drafting):
  `DAILY_PUBLISH_CAP = 12`, `MIN_PUBLISH_GAP_MINUTES = 20`, state in
  `facebook_publish_state.json` / `instagram_publish_state.json`.
- State at 2026-09-30: 115 entries — 31 published, 47 rejected, 37 pending.
  Facebook/Instagram publishing has therefore been exercised for real.
- Credentials in `.credentials/` (gitignored; repo has a public remote):
  `youtube_client_secret.json`, `youtube_token.json`, `meta.env` (long-lived
  Page token, scopes incl. `instagram_basic` + `instagram_content_publish`),
  `discord.env` (bot token + channel id).
- Instagram needs no public hosting (resumable upload straight to
  `rupload.facebook.com`). Facebook captions lead with the title because the
  API `title` is never shown in the Feed.
- `SOCIAL_UPLOAD_COMPLETE.md` is the original (stale) handoff, superseded by
  this section; safe to delete.

## Known gotchas — don't rediscover
- **Node titles, not IDs** in config `set:`; duplicate titles within one
  workflow silently resolve to whichever node is found first.
- **Workflows must be API-format** exports (`load_workflow()` rejects
  UI-format with nodes/links keys).
- **Pixaroma nodes** keep real values in JSON-encoded string fields — use
  `json_field:`. Seed uses native RandomNoise/`SEED_BASE`/`SEED_REFINE`, not
  Seed Pixaroma.
- **`CR Save Text To File`** never creates its directory (pre-created by
  `run_stage`), writes in cp1252 (so QwenVL is told to use plain ASCII), and
  reports no job outputs — `comfy_client.wait()` accepts `status.completed`
  / `status_str == "success"` and `run_stage()` reads the predicted path.
  CR Save never overwrites: it renames `_1`, `_2` on collision.
- **A `produces: prompt` stage never becomes `previous_file`**; its text
  overwrites `ctx[stage["sets"]]`. Don't remove that guard.
- **Windows encoding**: stdout/stderr reconfigured to UTF-8 at import time in
  `pipeline.py`; every config/ledger read/write uses `encoding="utf-8"`.
  Emoji node titles ("Video Combine 🎥🅥🅗🅢") are targeted by exact title.
- **QwenVL nodes must keep `keep_model_loaded: false`**; models unload
  between every stage (`free_vram_between_stages: true`, never turn off).
  A third sequential QwenVL call per job reproduced an earlier call's output
  verbatim, so video prompt + metadata are one combined call.
- **ComfyUI must run from the venv** (`C:\ComfyUI\venv`, transformers
  4.57.6), launched with cwd `C:/ComfyUI` and `PYTHONUTF8=1` — see the long
  comment in config.yaml `comfy_restart_command`. Using the system Python
  broke the FP8 Qwen model; launching from the wrong cwd broke Triton's
  ptxas lookup. Don't change without re-verifying QwenVL.
- **VHS_LoadVideo audio passthrough** spawned ffmpeg inside ComfyUI and
  failed ~100% on fresh clips, which is why audio is a separate stage and is
  attached at the interpolate stage via `LOAD_AUDIO`.
- **RIFE does not reliably honor `/interrupt`** — expect the current
  interpolation to finish when stopping a run.
- **`validate_workflow` enum matching is stricter than ComfyUI**
  (`scale_factor` must be float `1.0`); harmless elsewhere.
- Failure modes seen in the ledger: ComfyUI `execution_interrupted`/error
  traces during sampling (most), stage timeouts (`did not finish within
  1200s/300s`), dropped connections (WinError 10054), and one "ComfyUI
  refused the workflow" validation error. The last three jobs on 2026-09-29
  failed this way. Failures are logged and the run moves on.

## Working preferences
- No automated quality control: it renders and saves; the user reviews
  `FINAL/` and the Discord posts. Don't add scoring/filtering/auto-reject
  unless asked. (The camera/TITLE keyword retry is a mechanical safety net,
  not a quality judgment.)
- Prompt style: terse comma-separated concrete fragments for stills, with
  the Arri Alexa / Kodak Vision3 500T closing block, only practical warm
  sources, no people, no golden hour. Motion prompts: one camera move, rest
  "completely still", only ambient motion.
- Titles lead with curiosity/emotion, not scene description. Snow/ice/cold is
  the highest-performing territory.
- `grade_accent.json` in the output root controls a warm-accent grade trial
  (limited count, then a permanent small highlight lift via `after`). Used
  inside `grade_video()`.

## Status as of 2026-09-29 (last activity before this rewrite)
- Last real run ended 2026-09-29 ~17:09, mid-RIFE on job
  `20260929_164905_a74ca15d` (prompt index 22; `prompts_position.json` says
  next_index 23). Running `resume_production.bat` will roll back to index 22
  and regenerate it.
- Last good job finished 15:56 that day; the three jobs after it failed
  (two timeouts at 1200s, one interrupted KSampler). Cause not yet
  investigated.
- Recent tuning (2026-09-26 to 09-30): 2K SeedVR2 settings, LTX levels/2k
  backups, Flux LoRA 0.5, audio shaping, accent grade. Whether the 2K
  upscale settings were finalized is unconfirmed.
- 37 approvals still `pending` on Discord at last check.

## Open questions for the user (not recoverable from code)
- Was the 2K SeedVR2 setup (`03_seedvr2.json`, `workflows/test2k/`) accepted
  as final?
- Why did the last 3 jobs fail (timeout at 1200s looks like the video stage
  hanging or the upscale being slow)? Needs errors.log/run.log review.
- Is YouTube staying off permanently, or will a new channel be set up?
- What should happen to the 37 pending approvals and to
  `prompts/lucidscapes_prompts*.json` (batch 1, selected) — only batch2 is
  referenced by config.
- The old "size up QwenVL from 2B to 8B" task is moot while prompts come
  from the file; it only matters for `video_prompt` quality now.
