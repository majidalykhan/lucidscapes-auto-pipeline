"""
Talks to ComfyUI over its built-in web API.

Nothing clever here. It sends a workflow, waits for it to finish, and fetches
the file that came out. Every other file in this project uses this one.

ComfyUI must be running with its API reachable:
    python main.py --listen 127.0.0.1 --port 8188
"""

from __future__ import annotations

import copy
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests

try:
    import websocket  # package name: websocket-client
except ImportError:
    websocket = None  # live progress is optional; the run still works without it


class ComfyError(RuntimeError):
    """Something went wrong inside ComfyUI."""


class OutOfMemory(ComfyError):
    """ComfyUI ran out of VRAM. Recoverable: unload models and try again."""


OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "torch.outofmemoryerror",
    "allocation on device",
    "not enough memory",
)


class ComfyClient:
    def __init__(self, host="127.0.0.1", port=8188, input_dir="", output_dir="", timeout=30):
        self.base = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())
        self.timeout = timeout
        # ComfyUI's own input folder. Because everything runs on one machine we
        # copy files there directly, which is far more reliable than uploading
        # them over HTTP — especially for video, where different loader nodes
        # expect different upload endpoints.
        self.input_dir = Path(input_dir) if input_dir else None
        # ComfyUI's own output folder — the root every save node writes under,
        # no matter what folder or filename_prefix an individual node is
        # configured with. Knowing this root is what lets download() clean up
        # ComfyUI's original copy after ours is safely saved, so a file
        # doesn't end up duplicated in two places.
        self.output_dir = Path(output_dir) if output_dir else None

    # ---------- plumbing ----------

    def _get(self, path, **kw):
        r = requests.get(self.base + path, timeout=self.timeout, **kw)
        r.raise_for_status()
        return r

    def _post(self, path, **kw):
        r = requests.post(self.base + path, timeout=self.timeout, **kw)
        r.raise_for_status()
        return r

    def is_up(self) -> bool:
        try:
            self._get("/system_stats")
            return True
        except Exception:
            return False

    def vram(self) -> dict:
        """Free and total VRAM in GB, so you can see headroom before a stage."""
        try:
            dev = self.system_stats()["devices"][0]
            return {"free_gb": round(dev["vram_free"] / 1e9, 1),
                    "total_gb": round(dev["vram_total"] / 1e9, 1)}
        except Exception:
            return {}

    def system_stats(self) -> dict:
        return self._get("/system_stats").json()

    def queue_depth(self) -> int:
        q = self._get("/queue").json()
        return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))

    def interrupt(self) -> None:
        try:
            self._post("/interrupt")
        except Exception:
            pass

    def free_memory(self) -> None:
        """Unload models from VRAM.

        This is the single most important call in the whole project. You have
        four large models (Flux, MiniMax, SeedVR2, RIFE) and one 32 GB card.
        Unloading between every stage is what stops them stacking up until
        something runs out of memory.
        """
        try:
            self._post("/free", json={"unload_models": True, "free_memory": True})
        except Exception:
            pass  # older ComfyUI builds may not have /free; not fatal

    # ---------- workflow files ----------

    @staticmethod
    def load_workflow(path: str | Path) -> dict:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "nodes" in data and "links" in data:
            raise ComfyError(
                f"{Path(path).name} was saved with the normal Save button.\n"
                f"Re-export it with  Workflow -> Export (API)  and overwrite this file."
            )
        return data

    @staticmethod
    def find_by_title(graph: dict, title: str) -> str:
        for node_id, node in graph.items():
            if node.get("_meta", {}).get("title") == title:
                return node_id
        titles = sorted(n.get("_meta", {}).get("title", "?") for n in graph.values())
        raise ComfyError(
            f"No node is titled '{title}'.\n"
            f"Titles currently in this workflow: {titles}\n"
            f"Right-click the node in ComfyUI -> Title, rename it, then re-export."
        )

    @classmethod
    def set_input(cls, graph: dict, title: str, key: str, value: Any) -> None:
        node_id = cls.find_by_title(graph, title)
        node = graph[node_id]
        if key not in node["inputs"]:
            raise ComfyError(
                f"Node '{title}' is a {node['class_type']} and has no setting called '{key}'.\n"
                f"Its settings are: {sorted(node['inputs'])}"
            )
        node["inputs"][key] = value

    @classmethod
    def set_seed(cls, graph: dict, title: str, seed: int) -> None:
        node = graph[cls.find_by_title(graph, title)]
        for key in ("seed", "noise_seed"):
            if key in node["inputs"]:
                node["inputs"][key] = int(seed)
                return
        raise ComfyError(f"Node '{title}' has no seed setting.")

    @staticmethod
    def describe(graph: dict) -> str:
        """Print every node with its title and settings.

        Run this whenever a title error appears — it shows you exactly what is
        in the workflow so you can copy the right names into config.yaml.
        """
        lines = []
        for node_id, node in sorted(graph.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
            title = node.get("_meta", {}).get("title", "")
            lines.append(f"{node_id:>4}  {node['class_type']:<34} title = {title!r}")
            for k, v in node["inputs"].items():
                if not isinstance(v, list):  # lists are wires to other nodes
                    shown = (str(v)[:64] + "...") if len(str(v)) > 64 else v
                    lines.append(f"          .{k} = {shown}")
        return "\n".join(lines)

    # ---------- getting files in and out ----------

    def stage_input_file(self, path: str | Path, subfolder: str = "lucid") -> str:
        """Put a file where ComfyUI's loader nodes can see it.

        Returns the string a LoadImage / VHS_LoadVideo node expects, which is a
        path relative to ComfyUI's input folder.
        """
        path = Path(path)
        if self.input_dir:
            dest_dir = self.input_dir / subfolder
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_dir / path.name)
            return f"{subfolder}/{path.name}"
        # fallback: upload over HTTP (images only)
        with path.open("rb") as fh:
            r = self._post("/upload/image",
                           files={"image": (path.name, fh)},
                           data={"overwrite": "true", "subfolder": subfolder, "type": "input"})
        info = r.json()
        sub = info.get("subfolder", "")
        return f"{sub}/{info['name']}" if sub else info["name"]

    def download(self, ref: dict, dest: str | Path) -> Path:
        r = self._get("/view", params={
            "filename": ref["filename"],
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        })
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        self._cleanup_source(ref, dest)
        return dest

    def _cleanup_source(self, ref: dict, dest: Path) -> None:
        """Delete ComfyUI's own copy of a file once ours is safely saved.

        Every save node writes to ComfyUI's output folder no matter what its
        filename_prefix says — that write happens before we ever see the
        file, and we have no way to stop it. Without this, every stage leaves
        a second copy sitting wherever the workflow's save node pointed
        (a plain default folder, or a custom one like "Cozy Cinematic"), on
        top of the organized copy the pipeline just made. This removes that
        original so the pipeline's own output folder is the only place the
        file ends up.

        Best-effort only: a cleanup failure never fails the run — a leftover
        duplicate is a minor annoyance, refusing to continue over it is not
        an acceptable trade. But it used to be silent on top of that, and a
        real run left confirmed leftovers (Windows/antivirus briefly locking
        a file right after it's written is the usual cause — deletion
        retried a few lines below happens to dodge exactly that). Silent
        meant nobody could tell a leftover had happened short of manually
        diffing folders, so this now retries briefly and prints a visible
        warning if it still can't remove the source, instead of swallowing
        the failure with no trace.
        """
        if not self.output_dir or ref.get("type", "output") != "output":
            return
        source = self.output_dir / ref.get("subfolder", "") / ref["filename"]
        for attempt in range(3):
            try:
                if not source.exists() or source.resolve() == dest.resolve():
                    return
                source.unlink()
                return
            except Exception:
                if attempt < 2:
                    time.sleep(0.5)
        print(f"[comfy_client] could not remove ComfyUI's own copy of the file: {source}",
              flush=True)

    @staticmethod
    def collect_outputs(entry: dict) -> list[dict]:
        """Every file the job produced.

        Save nodes disagree about what to call their output: SaveImage says
        'images', Video Helper Suite says 'gifs', SaveVideo says 'videos'. So
        scan all of them rather than guessing.
        """
        files = []
        for node_out in entry.get("outputs", {}).values():
            for value in node_out.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "filename" in item:
                            files.append(item)
        return files

    # ---------- running ----------

    def submit(self, graph: dict) -> str:
        try:
            r = self._post("/prompt", json={"prompt": copy.deepcopy(graph),
                                            "client_id": self.client_id})
        except requests.HTTPError as e:
            body = e.response.text[:2000] if e.response is not None else str(e)
            if any(m in body.lower() for m in OOM_MARKERS):
                raise OutOfMemory(body) from e
            raise ComfyError(f"ComfyUI refused the workflow:\n{body}") from e
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, timeout: int, poll: float = 2.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            entry = self._get(f"/history/{prompt_id}").json().get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msg = json.dumps(status.get("messages", []))[:2000]
                    if any(m in msg.lower() for m in OOM_MARKERS):
                        raise OutOfMemory(msg)
                    raise ComfyError(msg)
                # A graph whose only output node writes a file itself (e.g.
                # Comfyroll's "CR Save Text To File") reports no "outputs" at
                # all — that key stays {} even after a clean run — so relying
                # on entry["outputs"] alone left this waiting for something
                # that was never coming, until the stage timed out. status
                # "completed" (or status_str "success") is ComfyUI's own
                # signal that execution finished; check that FIRST, and treat
                # a populated "outputs" as an earlier-arriving success signal
                # for graphs that do report through it.
                if status.get("completed") or status.get("status_str") == "success" \
                        or entry.get("outputs"):
                    return entry
            time.sleep(poll)
        raise ComfyError(f"job did not finish within {timeout}s")

    # ---------- live progress ----------

    def _print_progress(self, prompt_id: str, node_labels: dict, on_update, stop: threading.Event) -> None:
        """Report what ComfyUI is doing right now, as plain lines rather than
        an in-place bar drawn with carriage returns.

        The carriage-return trick only overwrites correctly on a terminal that
        supports it, live, with nothing else writing to the same stream. It
        does not survive being piped, logged to a file, or a race against the
        main thread's own print at the moment a stage finishes — which is
        exactly the dangling bar you saw. Plain timestamped lines have none of
        those failure modes: every terminal shows them correctly, and routing
        them through on_update means they land in run.log too.

        node_labels maps a node's numeric id to something readable (its title,
        or its type if it has no title), so the line says "running: KSampler"
        rather than a bare, meaningless number.

        Whether a node reports a step count depends on the node itself, not
        on which model or brand it belongs to: samplers (KSampler,
        SamplerCustomAdvanced, and similar) loop locally and report every
        step. Nodes with no internal loop — VAE decode, saving a file, a
        remote API call — only ever announce "started" and "finished," in any
        workflow, Flux included. For those this reports elapsed time instead,
        so a slow one doesn't look frozen.
        """
        if websocket is None:
            return
        log = on_update or (lambda text: print(text, flush=True))
        ws_url = self.base.replace("http://", "ws://").replace("https://", "wss://")
        try:
            ws = websocket.create_connection(f"{ws_url}/ws?clientId={self.client_id}", timeout=5)
        except Exception:
            return
        ws.settimeout(1.0)

        ELAPSED_EVERY = 10.0   # seconds between "still running" pings for silent nodes
        PROGRESS_EVERY = 4.0   # seconds between step updates on a noisy node

        current_node = None
        node_started_at = 0.0
        last_reported_at = 0.0

        while not stop.is_set():
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                if current_node is not None and time.time() - last_reported_at >= ELAPSED_EVERY:
                    label = node_labels.get(current_node, f"node {current_node}")
                    log(f"{label}: still running ({int(time.time() - node_started_at)}s, "
                        f"no step data from this node)")
                    last_reported_at = time.time()
                continue
            except Exception:
                break
            if isinstance(raw, (bytes, bytearray)):
                continue  # binary frames are preview images, not progress

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            data = msg.get("data", {})
            if data.get("prompt_id") not in (None, prompt_id):
                continue  # someone else's job running on the same server

            mtype = msg.get("type")
            if mtype == "executing" and data.get("prompt_id") == prompt_id:
                if data.get("node") is None:
                    break  # ComfyUI's own signal that this job is finished
                current_node = data["node"]
                node_started_at = last_reported_at = time.time()
                log(f"running: {node_labels.get(current_node, f'node {current_node}')}")
            elif mtype == "progress" and current_node is not None:
                value, maximum = data.get("value", 0), max(data.get("max", 1), 1)
                now = time.time()
                if value >= maximum or now - last_reported_at >= PROGRESS_EVERY:
                    label = node_labels.get(current_node, f"node {current_node}")
                    log(f"{label}: step {value}/{maximum} ({int(100 * value / maximum)}%)")
                    last_reported_at = now

        try:
            ws.close()
        except Exception:
            pass

    def run(self, graph: dict, timeout: int = 3600, show_progress: bool = False, on_update=None) -> list[dict]:
        prompt_id = self.submit(graph)
        stop = threading.Event()
        thread = None
        if show_progress:
            node_labels = {nid: (node.get("_meta", {}).get("title") or node.get("class_type", nid))
                          for nid, node in graph.items()}
            thread = threading.Thread(target=self._print_progress,
                                      args=(prompt_id, node_labels, on_update, stop), daemon=True)
            thread.start()
        try:
            entry = self.wait(prompt_id, timeout)
        finally:
            stop.set()
            if thread:
                thread.join(timeout=2)
        return self.collect_outputs(entry)


if __name__ == "__main__":
    import sys
    print(ComfyClient.describe(ComfyClient.load_workflow(sys.argv[1])))
