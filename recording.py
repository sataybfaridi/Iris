"""Incident recording for IRIS (the dashcam feature) -- WITH sound.
 
This records the webcam AND the microphone into one .mp4 using ffmpeg, and
saves an entry in recordings.json. (OpenCV, which we used before, can only do
silent video -- ffmpeg records picture + audio together.)
 
Install ffmpeg once:
    macOS:  brew install ffmpeg
    Pi:     sudo apt install ffmpeg
 
How you'd use it:
    recording.start_recording(trigger="voice")   # user said "record"
    ... time passes ...
    recording.stop_recording()                    # stops + saves the clip
 
Picking the camera/mic (only if the default is wrong):
    See your devices on macOS:
        ffmpeg -f avfoundation -list_devices true -i ""
    Then set, e.g.:
        IRIS_AV_DEVICE="1:0"     # "video:audio" index
    On the Pi (Linux):
        IRIS_V4L2_DEV=/dev/video0   IRIS_ALSA_DEV=default
 
Files (same place the device/dashboard read from):
    data/recordings.json            list of clips (metadata)
    data/recordings/clip_*.mp4      the video+audio files
"""
 
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
 
logger = logging.getLogger(__name__)
 
# Everything lives in a "data" folder next to this file -- the same place the
# contacts and account live -- so the whole device has ONE data folder.
_DEFAULT_IRIS_DIR = Path(__file__).resolve().parent / "data"
IRIS_DIR = Path(os.getenv("IRIS_DIR", str(_DEFAULT_IRIS_DIR)))
RECORDINGS_PATH = Path(os.getenv("IRIS_RECORDINGS_PATH", str(IRIS_DIR / "recordings.json")))
VIDEO_DIR = Path(os.getenv("IRIS_VIDEO_DIR", str(IRIS_DIR / "recordings")))
FFMPEG_LOG = IRIS_DIR / "recording_ffmpeg.log"
 
FPS = 30  # frames per second to record
 
# State for the one recording that can be in progress.
_lock = threading.Lock()
_state = {
    "recording": False, "trigger": "", "started": 0.0,
    "proc": None, "path": None, "mock": False,
}
 
 
def _load() -> list:
    try:
        return json.loads(RECORDINGS_PATH.read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as error:
        logger.warning("Could not read recordings file: %s", error)
        return []
 
 
def _save(items: list) -> None:
    RECORDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORDINGS_PATH.write_text(json.dumps(items, indent=2))
 
 
def is_recording() -> bool:
    return _state["recording"]
 
 
def _ffmpeg_exe() -> str | None:
    """Find an ffmpeg binary: IRIS_FFMPEG override, then one on PATH (e.g. from
    Homebrew), then the one bundled by the pip package `imageio-ffmpeg` (so you
    don't need Homebrew). Returns None if none is available."""
    override = os.getenv("IRIS_FFMPEG")
    if override:
        return override
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
 
 
# Virtual/loopback "mics" that show up from other apps but carry no real sound.
# We skip these so we don't record silence (ZoomAudioDevice was the culprit).
_FAKE_MIC_HINTS = ("zoom", "teams", "aggregate", "multi-output", "blackhole",
                   "loopback", "soundflower", "virtual", "cable", "webex", "obs")
_mac_mic_cache = None
 
 
def _detect_mac_mic(ffmpeg: str):
    """Ask macOS for its audio devices and return the name of the REAL mic,
    skipping virtual ones. Falls back to index '0' if nothing better is found."""
    global _mac_mic_cache
    if _mac_mic_cache is not None:
        return _mac_mic_cache
    try:
        out = subprocess.run(
            [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=10,
        ).stderr
    except Exception:
        _mac_mic_cache = "0"
        return _mac_mic_cache
 
    audio, in_audio = [], False
    for line in out.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True; continue
        if "AVFoundation video devices" in line:
            in_audio = False; continue
        if in_audio:
            m = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
            if m:
                audio.append(m.group(2))
 
    def looks_fake(name): return any(h in name.lower() for h in _FAKE_MIC_HINTS)
    # 1) a real built-in mic by name
    for name in audio:
        low = name.lower()
        if ("microphone" in low or "built-in" in low or "macbook" in low) and not looks_fake(name):
            _mac_mic_cache = name; return name
    # 2) any non-virtual device
    for name in audio:
        if not looks_fake(name):
            _mac_mic_cache = name; return name
    # 3) give up -> first device, or index 0
    _mac_mic_cache = audio[0] if audio else "0"
    return _mac_mic_cache
 
 
def _ffmpeg_command(path: Path, ffmpeg: str) -> list:
    """Build the ffmpeg command that grabs camera + mic into an mp4 with audio.
    The capture inputs differ by operating system."""
    common_out = [
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", str(path),
    ]
    if platform.system() == "Darwin":                       # macOS
        # Use IRIS_AV_DEVICE if set; otherwise pick the real mic automatically.
        device = os.getenv("IRIS_AV_DEVICE")
        if not device:
            video = os.getenv("IRIS_AV_VIDEO", "0")         # camera index
            device = f"{video}:{_detect_mac_mic(ffmpeg)}"   # "0:MacBook Air Microphone"
        return ([ffmpeg, "-y", "-f", "avfoundation",
                 "-framerate", str(FPS), "-i", device] + common_out)
    # Linux / Raspberry Pi
    video = os.getenv("IRIS_V4L2_DEV", "/dev/video0")
    audio = os.getenv("IRIS_ALSA_DEV", "default")
    return ([ffmpeg, "-y",
             "-f", "v4l2", "-framerate", str(FPS), "-i", video,
             "-f", "alsa", "-i", audio] + common_out)
 
 
def start_recording(trigger: str = "voice", use_camera: bool = True) -> dict:
    """Begin recording webcam + mic with ffmpeg. trigger is 'voice'/'button'/'auto'.
 
    Fails honestly (no fake recording) if ffmpeg isn't installed or the camera
    can't be opened. use_camera=False runs a deliberate no-capture mock.
    """
    with _lock:
        if _state["recording"]:
            return {"status": "already", "message": "Already recording."}
 
        if not use_camera:
            _state.update(recording=True, trigger=trigger, started=time.monotonic(),
                          path=None, mock=True, proc=None)
            print("[Recording] Started (mock, no capture).", flush=True)
            return {"status": "started", "mock": True, "message": "Recording started (mock)."}
 
        ffmpeg = _ffmpeg_exe()
        if ffmpeg is None:
            return {"status": "no_ffmpeg",
                    "message": "ffmpeg isn't installed, so I can't record with sound. "
                               "Install it with pip install imageio-ffmpeg."}
 
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        IRIS_DIR.mkdir(parents=True, exist_ok=True)
        path = VIDEO_DIR / f"clip_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        try:
            log = open(FFMPEG_LOG, "wb")
            proc = subprocess.Popen(
                _ffmpeg_command(path, ffmpeg),
                stdin=subprocess.PIPE, stdout=log, stderr=log,
            )
        except Exception as error:
            return {"status": "error", "message": f"Could not start ffmpeg: {error!r}"}
 
        # Give ffmpeg a moment to actually open the camera/mic. If it died, the
        # device is busy or the device index is wrong -- fail honestly.
        time.sleep(1.2)
        if proc.poll() is not None:
            return {"status": "no_camera",
                    "message": "Couldn't open the camera or microphone. The camera may be "
                               "in use by the vision feature, or the device index is wrong "
                               f"(see {FFMPEG_LOG})."}
 
        _state.update(recording=True, trigger=trigger, started=time.monotonic(),
                      path=path, mock=False, proc=proc)
        print("[Recording] Started (webcam + mic).", flush=True)
        return {"status": "started", "mock": False, "message": "Recording started."}
 
 
def stop_recording() -> dict:
    """Stop recording and save the clip (video + audio) + its metadata."""
    with _lock:
        if not _state["recording"]:
            return {"status": "idle", "message": "Nothing is being recorded right now."}
        proc = _state["proc"]
        started = _state["started"]
        path = _state["path"]
        trigger = _state["trigger"]
 
    if proc is not None:  # tell ffmpeg to stop and finalize the mp4 (outside the lock)
        try:
            proc.communicate(input=b"q", timeout=10)   # 'q' = quit cleanly
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
 
    with _lock:
        seconds = max(1, int(time.monotonic() - started))
        clip = {
            "title": f"Clip {datetime.now():%b %d, %I:%M %p}",
            "when": datetime.now().isoformat(timespec="seconds"),
            "length": f"{seconds // 60}:{seconds % 60:02d}",
            "trigger": trigger,
            "file": str(path) if path else "",
        }
        items = _load()
        items.insert(0, clip)  # newest first
        _save(items)
        where = clip["file"] or "(mock, no file)"
        _state.update(recording=False, trigger="", started=0.0, proc=None,
                      path=None, mock=False)
        print(f"[Recording] Saved {clip['length']} -> {where}", flush=True)
 
    cleanup_old_recordings()        # keep the device from filling up
    return {"status": "saved", "message": f"Saved a {clip['length']} recording.", "clip": clip}
 
 
# Don't let recordings use more than this much of the device. Oldest clips are
# deleted first once the total goes over. Override with IRIS_MAX_RECORDINGS_GB.
MAX_RECORDINGS_BYTES = int(float(os.getenv("IRIS_MAX_RECORDINGS_GB", "30")) * 1024 ** 3)
 
 
def cleanup_old_recordings(max_bytes: int = MAX_RECORDINGS_BYTES) -> list:
    """Delete the oldest clips until recordings fit under max_bytes, and drop
    metadata entries whose files no longer exist. Returns the names removed."""
    removed = []
    try:
        files = sorted(VIDEO_DIR.glob("clip_*.mp4"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return removed
    total = sum(p.stat().st_size for p in files if p.exists())
    i = 0
    while total > max_bytes and i < len(files):
        old = files[i]; i += 1
        try:
            total -= old.stat().st_size
            old.unlink()
            removed.append(old.name)
            print(f"[Recording] Cleaned up old clip: {old.name}", flush=True)
        except OSError:
            pass
    # Keep recordings.json in sync: drop entries whose file is gone.
    with _lock:
        items = _load()
        kept = [c for c in items if (not c.get("file")) or Path(c["file"]).exists()]
        if len(kept) != len(items):
            _save(kept)
    return removed
 
 
def list_recordings() -> list:
    return _load()
 
 
def delete_recording(name: str) -> dict:
    """Delete one clip by its file name (e.g. clip_20261116_033640.mp4).
    Removes the video file and its entry in recordings.json."""
    base = os.path.basename(name or "")
    if not base:
        return {"status": "error", "message": "No recording given."}
    target = (VIDEO_DIR / base)
    try:
        resolved = target.resolve()
        if VIDEO_DIR.resolve() not in resolved.parents:   # stay inside the folder
            return {"status": "error", "message": "That isn't a valid recording."}
    except Exception:
        resolved = target
    try:
        if resolved.exists():
            resolved.unlink()
    except OSError as error:
        return {"status": "error", "message": f"Couldn't delete it: {error}"}
    with _lock:
        items = _load()
        kept = [c for c in items if os.path.basename(c.get("file", "")) != base]
        _save(kept)
    print(f"[Recording] Deleted {base}", flush=True)
    return {"status": "deleted", "message": "Recording deleted."}
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(start_recording(trigger="voice"))   # records webcam + mic if ffmpeg is installed
    time.sleep(3)
    print(stop_recording())
    print("All recordings:", list_recordings())
 