"""Function-based Gemini Live API transport skeleton."""
 
import array
import asyncio
import math
import os
import subprocess
import sys
import time
import webbrowser
from collections.abc import Awaitable, Callable
from typing import Any
 
from config import DeploymentMode, Settings
from vision_tools import (
    VisionState,
    build_camera_tools,
    build_system_instruction,
    handle_tool_call,
    save_debug_frame,
)
from interfaces import (
    AUDIO_OUTPUT_SAMPLE_RATE_HZ,
    VIDEO_FRAMES_PER_SECOND,
    AudioOutputFrame,
    Button,
    CompanionPeripherals,
)
 
# The emergency alert file (emergency_calls.py, in the project root). Calling
# its call_emergency_contacts() rings your saved contacts in priority order.
import emergency_calls
# The dashcam recorder (recording.py, in the project root). Saying "start
# recording" / "stop recording" drives it.
import recording
 
 
# ============================================================
# Emergency button settings
# ============================================================
 
# Hold the emergency button at least this long before the alert fires. A
# quick accidental tap does nothing.
EMERGENCY_HOLD_SECONDS = 5.0
 
# Where to say the user is. This is a placeholder for mock testing; the real
# device will fill this in from GPS.
EMERGENCY_LOCATION = "41.8006, -87.7845 (Morton Grove, IL) (mock)"
 
 
# Added to Gemini's system instruction so it stops saying "I can't record."
# The DEVICE does the recording itself; Gemini just needs to acknowledge.
RECORDING_INSTRUCTION = (
    "The device controls video recording by itself and announces the real "
    "result out loud on its own (for example 'Recording started', 'Recording "
    "saved', or an error if the camera isn't available). You do NOT control "
    "recording and you do NOT know whether it actually worked. Therefore never "
    "claim that recording started, stopped, saved, or failed, and never say "
    "things like 'okay, recording'. If the user asks to start or stop "
    "recording, do not comment on the recording itself -- let the device "
    "announce it. You may answer any other part of what they said."
)
 
 
def _with_recording_note(instruction):
    """Append the recording note to the system instruction, whether it's a
    plain string or a Content object (so it works either way)."""
    try:
        if isinstance(instruction, str):
            return instruction + "\n\n" + RECORDING_INSTRUCTION
        from google.genai import types
        if hasattr(instruction, "parts") and instruction.parts is not None:
            instruction.parts.append(types.Part(text="\n\n" + RECORDING_INSTRUCTION))
            return instruction
    except Exception:
        pass
    return instruction          # worst case: unchanged, never crashes
 
 
def _is_inactivity_close(error: BaseException) -> bool:
    """True when Gemini closed the session just because it was idle. That's
    expected (the connection is meant to close after inactivity), so we
    reconnect instead of crashing."""
    text = f"{error}".lower()
    return "inactivity" in text or "1008" in text
 
 
async def _wait_for_talk_press(button: Button) -> None:
    """Block until the Talk button is pressed. Used to wake a closed session:
    after Gemini times out, we don't reconnect until the user presses Talk, so
    we're not holding a connection open while idle. Reads the button's pressed
    state (doesn't consume its event queue, so the press still reaches the
    normal handler once we reconnect)."""
    was_pressed = bool(getattr(button, "_pressed", False))
    while True:
        now_pressed = bool(getattr(button, "_pressed", False))
        if now_pressed and not was_pressed:      # a fresh press
            return
        was_pressed = now_pressed
        await asyncio.sleep(0.05)
 
 
async def run_gemini_live(
    settings: Settings, peripherals: CompanionPeripherals
) -> None:
    """Run the device and keep a Gemini Live session connected, reconnecting
    automatically whenever Gemini closes it for inactivity.
 
    The device tasks (dashboard, status, emergency button, setup mode) run the
    WHOLE time -- independent of Gemini -- so the SOS button and monitoring keep
    working even if Gemini or the internet drops.
    """
    from google import genai
    from google.genai import types
 
    client = genai.Client(api_key=settings.gemini_api_key)
    state = VisionState()
    setup_state = SetupState()
 
    # ---- always-on device tasks (survive Gemini reconnects) ----
    device_tasks = [asyncio.create_task(write_status_loop(peripherals, setup_state))]
    emergency_button = getattr(peripherals, "emergency_button", None)
    if emergency_button is not None:
        device_tasks.append(asyncio.create_task(listen_for_emergency_hold(emergency_button)))
    talk_button = getattr(peripherals, "talk_button", None)
    camera_button = getattr(peripherals, "camera_button", None)
    if talk_button is not None and camera_button is not None:
        device_tasks.append(asyncio.create_task(
            listen_for_setup_combo(talk_button, camera_button, setup_state)))
 
    await start_web_server(setup_state)
 
    def _live_config():
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            # Let Gemini see pictures at full quality (small label text).
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            # Camera tools + rules. The laptop webcam faces you, so left and
            # right are flipped; the Pi camera points away like a phone.
            system_instruction=_with_recording_note(
                build_system_instruction(
                    camera_faces_user=settings.mode is DeploymentMode.MOCK
                )
            ),
            tools=[build_camera_tools(types)],
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
 
    try:
        while True:                       # reconnect loop
            try:
                async with client.aio.live.connect(
                    model=settings.gemini_live_model, config=_live_config()
                ) as session:
                    await stream_live_session(peripherals, session, state, setup_state)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if _is_inactivity_close(error):
                    # Expected idle timeout: close the connection and WAIT for
                    # the user to press Talk before reconnecting (so we don't
                    # hold Gemini open while nothing is happening).
                    if talk_button is not None:
                        print("[Live] Iris went idle and disconnected. Press Talk to wake it.",
                              flush=True)
                        await _wait_for_talk_press(talk_button)
                        print("[Live] Talk pressed -- reconnecting to Gemini...", flush=True)
                    else:
                        await asyncio.sleep(1.0)
                    continue
                raise                     # a real error -> let it surface
    finally:
        # Stop the dashboard website and the always-on device tasks.
        server = setup_state.server
        if server is not None and server.poll() is None:
            try:
                server.terminate()
            except Exception:
                pass
        for task in device_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*device_tasks, return_exceptions=True)
 
 
async def stream_live_session(
    peripherals: CompanionPeripherals,
    session: Any,
    state: "VisionState | None" = None,
    setup_state: "SetupState | None" = None,
) -> None:
    """Run the Gemini-dependent tasks (responses + talk/camera) for ONE session.
    Returns/raises when the session ends; run_gemini_live reconnects if it was
    just an inactivity timeout."""
    if state is None:
        state = VisionState()
    tasks = (
        asyncio.create_task(receive_live_responses(session, peripherals, state)),
        asyncio.create_task(
            listen_for_button_events(peripherals, session, state, setup_state)
        ),
    )
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()                 # re-raises the real reason (e.g. inactivity close)
        raise RuntimeError("Gemini Live interaction task(s) ended unexpectedly.")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
 
 
# ============================================================
# Emergency button: hold to fire the alert
# ============================================================
 
async def _fire_emergency() -> None:
    """Call the saved emergency contacts (runs the blocking call off-loop)."""
    print("[Emergency] Button held — firing emergency alert.", flush=True)
    try:
        result = await asyncio.to_thread(
            emergency_calls.call_emergency_contacts,
            EMERGENCY_LOCATION,      # location
            "emergency button",      # reason
            None,                    # mock=None -> decided by IRIS_CALLS_MODE in .env
            False,                   # interactive=False (don't block on input())
        )
        print(f"[Emergency] {result.get('message', result)}", flush=True)
    except Exception as error:  # an alert failure must not crash IRIS
        print(f"[Emergency] Alert failed: {error!r}", flush=True)
 
 
async def listen_for_emergency_hold(
    button: Button, hold_seconds: float = EMERGENCY_HOLD_SECONDS
) -> None:
    """Fire the emergency alert when the button is held for hold_seconds.
 
    A quick tap does nothing: releasing before the time is up cancels it.
    """
    pressed_at: float | None = None
    async for event in button.events():
        if event.pressed:
            # Button went down: start the hold timer.
            pressed_at = time.monotonic()
        else:
            # Button came up: fire only if it was held long enough.
            if pressed_at is not None:
                held = time.monotonic() - pressed_at
                pressed_at = None
                if held >= hold_seconds:
                    await _fire_emergency()
                else:
                    remaining = hold_seconds - held
                    print(
                        f"[Emergency] Released too soon "
                        f"(hold {remaining:.1f}s longer to trigger).",
                        flush=True,
                    )
 
 
# ============================================================
# Setup mode: hold Camera + Talk to open the contacts website
# ============================================================
 
# Hold both buttons together this long before setup mode opens. Long enough
# that holding just one (talk or camera) on its own never triggers it.
SETUP_HOLD_SECONDS = 2.0
SETUP_PORT = os.getenv("IRIS_WEB_PORT", "8080")
SETUP_URL = f"http://localhost:{SETUP_PORT}"
 
# When a button is pressed, wait this long to see if the OTHER setup button is
# also going down. If it is, this press is part of the Camera+Talk setup
# gesture, so talk/camera do NOT start. A bit forgiving so a slightly
# staggered two-button press still counts as the gesture, not a talk.
SETUP_COMBO_WINDOW = 0.35
 
 
class SetupState:
    """Shared device state.
 
    `active` = setup mode, which decides whether the emergency contacts can be
    EDITED. The dashboard website itself runs the whole time the device is on
    (so family can always see status and recordings); only editing is gated.
    `server` is that always-on website process.
    """
 
    def __init__(self) -> None:
        self.active = False
        self.server: subprocess.Popen | None = None
 
 
# One announcement at a time (so two voices never overlap), and don't repeat
# the same line within a few seconds (so a few quick presses don't stack up).
_announce_lock = asyncio.Lock()
_announce_recent: dict[str, float] = {}
_ANNOUNCE_REPEAT_GAP = 4.0
 
 
async def _announce(text: str) -> None:
    """Say something out loud (and print it), one voice at a time.
 
    Uses macOS 'say' for the laptop mock; on the real device this is where the
    Pi's text-to-speech would go."""
    now = time.monotonic()
    # Drop a repeat of the exact same line heard a moment ago.
    if now - _announce_recent.get(text, 0.0) < _ANNOUNCE_REPEAT_GAP:
        return
    _announce_recent[text] = now
    print(f"[Setup] {text}", flush=True)
    # Hold the lock so a second announcement waits instead of talking over this
    # one -- you never get two 'say' voices at the same time.
    async with _announce_lock:
        try:
            if sys.platform == "darwin":
                await asyncio.to_thread(
                    subprocess.run, ["say", text],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass        # a missing TTS must never crash IRIS
 
 
async def start_web_server(setup_state: SetupState) -> None:
    """Start the dashboard website once, at boot, and keep it running the whole
    time the device is on. Monitoring (status + recordings) is therefore always
    available; editing contacts is gated separately by setup mode."""
    if setup_state.server is not None and setup_state.server.poll() is None:
        return
    try:
        env = dict(os.environ, IRIS_WEB_PORT=SETUP_PORT)
        setup_state.server = await asyncio.to_thread(
            subprocess.Popen, [sys.executable, "web_server.py"], env=env
        )
        await asyncio.sleep(1.0)
        print(f"[Dashboard] Monitoring website running at {SETUP_URL}", flush=True)
    except Exception as error:
        print(f"[Dashboard] Could not start the website: {error!r}", flush=True)
 
 
async def _enable_editing(setup_state: SetupState) -> None:
    """Setup mode ON: allow editing contacts, open the page, announce."""
    setup_state.active = True
    try:
        await asyncio.to_thread(webbrowser.open, SETUP_URL)
    except Exception:
        pass
    await _announce(
        "Setup mode is on. You can now edit the emergency contacts. Hold the "
        "camera and talk buttons together again to leave setup mode."
    )
 
 
async def _disable_editing(setup_state: SetupState) -> None:
    """Setup mode OFF: contacts go back to view-only. The dashboard stays up."""
    setup_state.active = False
    await _announce("Setup mode is off. The contacts are locked again.")
 
 
# ---- Live device status the dashboard reads ------------------------------
 
STATUS_PATH = emergency_calls.CONTACTS_PATH.parent / "status.json"
 
 
def _battery_percent():
    """Real battery percent (needs psutil), or None if it can't be read."""
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is not None:
            return int(battery.percent)
    except Exception:
        pass
    return None
 
 
async def write_status_loop(
    peripherals: CompanionPeripherals, setup_state: SetupState
) -> None:
    """Write the REAL device status to status.json a few times a second, so the
    dashboard shows live values (camera, recording, battery) instead of fakes."""
    import json
    while True:
        status = {
            "camera_on": bool(getattr(peripherals.camera, "is_on", False)),
            "recording": recording.is_recording(),
            "battery": _battery_percent(),
            "editing_allowed": setup_state.active,
            "updated": time.time(),
        }
        try:
            STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATUS_PATH.write_text(json.dumps(status))
        except OSError:
            pass
        await asyncio.sleep(1.5)
 
 
async def listen_for_setup_combo(
    talk_button: Button,
    camera_button: Button,
    setup_state: SetupState,
    hold_seconds: float = SETUP_HOLD_SECONDS,
) -> None:
    """Hold Talk + Camera together to TOGGLE setup mode on/off.
 
    Both buttons are already watched by the talk/camera handlers, so instead of
    consuming their events here (which would steal them), we just read their
    current pressed state. Each full hold toggles once; you have to release
    before it will toggle again.
    """
    held_since: float | None = None
    fired = False
    while True:
        both_down = (
            getattr(talk_button, "_pressed", False)
            and getattr(camera_button, "_pressed", False)
        )
        if both_down:
            if held_since is None:
                held_since = time.monotonic()
            elif not fired and time.monotonic() - held_since >= hold_seconds:
                fired = True
                if setup_state.active:
                    await _disable_editing(setup_state)
                else:
                    await _enable_editing(setup_state)
        else:
            held_since = None
            fired = False                 # released: ready to toggle again
        await asyncio.sleep(0.1)
 
 
# ============================================================
# Voice recording command ("start recording" / "stop recording")
# ============================================================
 
# What the device does on its own when it hears the recording command. The
# device runs the recording itself; Gemini is only told (via the system
# instruction) to briefly acknowledge, not to claim it can't record.
 
async def _maybe_handle_recording_command(text: str) -> None:
    """Start/stop the dashcam when the user says so, and announce the TRUE
    result in the device's own voice. The device is the source of truth --
    Gemini is told not to claim anything about recording."""
    low = text.lower()
    if "start recording" in low:
        if recording.is_recording():
            return                         # already going; nothing to announce
        result = await asyncio.to_thread(recording.start_recording, "voice")
        print(f"[Recording] (voice) {result.get('message', result)}", flush=True)
        if result.get("status") == "started":
            await _announce("Recording started.")
        else:
            # Honest failure -- say what actually went wrong.
            await _announce(result.get("message", "I couldn't start recording."))
    elif "stop recording" in low or "stop the recording" in low:
        if not recording.is_recording():
            return
        result = await asyncio.to_thread(recording.stop_recording)
        print(f"[Recording] (voice) {result.get('message', result)}", flush=True)
        if result.get("status") == "saved":
            await _announce("Recording saved.")
        else:
            await _announce(result.get("message", "I couldn't save the recording."))
 
 
async def send_audio_frames(
    peripherals: CompanionPeripherals, session: Any
) -> None:
    """Forward 16 kHz mono PCM microphone frames to the Live session."""
    # Import the SDK type here so importing this module does not require an
    # installed SDK until a Live session is actually used.
    from google.genai import types
 
    # The peripheral implementation owns microphone capture and yields frames
    # that already match the agreed mono, 16-bit, 16 kHz PCM contract.
    async for frame in peripherals.audio_input.frames():
        # Live API expects raw PCM bytes with the input sample rate in the MIME
        # type; the SDK wraps the bytes as a Blob for its WebSocket message.
        audio = types.Blob(
            data=frame.pcm,
            mime_type="audio/pcm;rate=16000",
        )
 
        # Send each captured frame immediately so the cloud stream stays live.
        await session.send_realtime_input(audio=audio)
 
 
# False = Gemini gets ONE fresh picture when you finish asking (plus the
# zoom/refocus pictures), not a nonstop stream. With a nonstop stream Gemini
# tends to latch onto an old frame. One picture per question is clearer,
# cheaper and more private. The camera itself stays running so it's warm.
STREAM_CAMERA_FRAMES = False
 
 
async def keep_camera_running(peripherals: CompanionPeripherals) -> None:
    """Keep the camera on (and warm) without sending anything to Gemini."""
    async for _ in peripherals.camera.frames():
        await asyncio.sleep(1 / VIDEO_FRAMES_PER_SECOND)
 
 
async def send_video_frames(
    peripherals: CompanionPeripherals, session: Any
) -> None:
    """Forward JPEG camera frames to the Live session at no more than 1 FPS."""
    from google.genai import types
 
    async for frame in peripherals.camera.frames():
        video = types.Blob(data=frame.jpeg, mime_type="image/jpeg")
        await session.send_realtime_input(video=video)
        await asyncio.sleep(1 / VIDEO_FRAMES_PER_SECOND)
 
 
async def receive_live_responses(
    session: Any,
    peripherals: CompanionPeripherals,
    state: VisionState | None = None,
) -> None:
    """Receive model events independently from speaker playback."""
    audio_queue: asyncio.Queue[AudioOutputFrame] = asyncio.Queue()
    tasks = (
        asyncio.create_task(
            receive_live_events(session, peripherals, audio_queue, state),
            name="gemini-live-receiver",
        ),
        asyncio.create_task(
            play_queued_audio(peripherals, audio_queue),
            name="gemini-live-audio-player",
        ),
    )
    try:
        done, _ = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            task.result()
        finished_names = ", ".join(task.get_name() for task in done)
        raise RuntimeError(
            "Gemini Live response task(s) ended unexpectedly: "
            f"{finished_names}."
        )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
 
 
async def receive_live_events(
    session: Any,
    peripherals: CompanionPeripherals,
    audio_queue: asyncio.Queue[AudioOutputFrame],
    state: VisionState | None = None,
) -> None:
    """Receive complete model turns continuously and enqueue their audio."""
    if state is None:
        state = VisionState()
    tool_tasks: set[asyncio.Task] = set()
    speaking = False
    while True:
        # The SDK receive iterator finishes after one complete model turn.
        # Start another iterator so the WebSocket remains ready for the next
        # push-to-talk interaction.
        received_response = False
        async for response in session.receive():
            received_response = True
 
            # Gemini asked to zoom/refocus. Run it in the background so we
            # keep receiving (camera work can take a second or two).
            tool_call = getattr(response, "tool_call", None)
            if tool_call is not None:
                task = asyncio.create_task(
                    handle_tool_call(tool_call, session, peripherals.camera, state)
                )
                tool_tasks.add(task)
                task.add_done_callback(_finish_tool_task(tool_tasks))
                continue
 
            speaking = await handle_live_response(
                response,
                peripherals,
                audio_queue,
                speaking=speaking,
            )
 
        if not received_response:
            raise RuntimeError(
                "Gemini Live response stream ended without a response."
            )
 
 
def _finish_tool_task(tool_tasks: set):
    def done(task: asyncio.Task) -> None:
        tool_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            print(f"[Vision] Camera tool failed: {task.exception()}", flush=True)
 
    return done
 
 
async def play_queued_audio(
    peripherals: CompanionPeripherals,
    audio_queue: asyncio.Queue[AudioOutputFrame],
) -> None:
    """Play model audio in a separate task so events remain responsive."""
    while True:
        frame = await audio_queue.get()
        try:
            await peripherals.audio_output.play(frame)
        finally:
            audio_queue.task_done()
 
 
async def handle_live_response(
    response: Any,
    peripherals: CompanionPeripherals,
    audio_queue: asyncio.Queue[AudioOutputFrame],
    speaking: bool = False,
) -> bool:
    """Queue PCM chunks, or flush queued/current audio on interruption."""
    server_content = response.server_content
    if server_content is None:
        return speaking
 
    heard = getattr(server_content, "input_transcription", None)
    if heard is not None and getattr(heard, "text", None):
        print(f"[You] {heard.text}", flush=True)
        # The device handles recording itself when it hears the command.
        await _maybe_handle_recording_command(heard.text)
    said = getattr(server_content, "output_transcription", None)
    if said is not None and getattr(said, "text", None):
        print(f"[Iris] {said.text}", flush=True)
 
    if server_content.interrupted:
        while True:
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                audio_queue.task_done()
        await peripherals.audio_output.interrupt()
        if speaking:
            print("[Live] Interrupted: stopped Gemini playback.", flush=True)
        speaking = False
 
    model_turn = server_content.model_turn
    if model_turn is not None:
        for part in model_turn.parts or ():
            inline_data = part.inline_data
            if inline_data is None:
                continue
 
            mime_type = inline_data.mime_type or ""
            if not mime_type.startswith("audio/pcm"):
                continue
            if f"rate={AUDIO_OUTPUT_SAMPLE_RATE_HZ}" not in mime_type:
                raise ValueError(
                    "Gemini audio must be 24 kHz PCM; received "
                    f"{mime_type!r}."
                )
            if not isinstance(inline_data.data, bytes):
                raise TypeError("Gemini PCM audio data must be bytes.")
 
            if not speaking:
                print("[Live] Speaking: playing Gemini audio.", flush=True)
                speaking = True
 
            audio_queue.put_nowait(AudioOutputFrame(pcm=inline_data.data))
 
    if speaking and server_content.turn_complete:
        print("[Live] Gemini finished speaking.", flush=True)
        speaking = False
 
    return speaking
 
 
# Told to Gemini at the start of every question, so it always knows if it
# has a live picture. Without this, when the camera is off Gemini keeps
# describing old pictures it still remembers.
CAMERA_ON_NOTE = (
    "[Camera note: the camera is ON. Pictures sent during this question are "
    "live. Answer only from them.]"
)
CAMERA_OFF_NOTE = (
    "[Camera note: the camera is OFF. You have NO current picture. Do not "
    "describe anything from earlier pictures. If this question needs the "
    "camera, ask the user to press the camera button.]"
)
 
# Taps shorter than this get padded with silence (see end_talking).
MIN_TALK_S = 1.0
# 0.5 s of silence: 16 kHz, mono, 16-bit = 2 bytes per sample.
SILENCE_PCM = bytes(16000 * 2 // 2)
 
 
async def listen_for_button_events(
    peripherals: CompanionPeripherals,
    session: Any,
    state: VisionState | None = None,
    setup_state: "SetupState | None" = None,
) -> None:
    """Hold Space to talk. Press C once to turn the camera on, again to turn it off."""
    from google.genai import types
 
    if state is None:
        state = VisionState()
    background: set[asyncio.Task] = set()
 
    question_streamer: asyncio.Task | None = None
    talk_started = 0.0
    told_camera_on = False
 
    async def tell_camera_on() -> None:
        """Camera turned on mid-question: tell Gemini before the first picture."""
        nonlocal told_camera_on
        if not told_camera_on:
            told_camera_on = True
            await session.send_realtime_input(text=CAMERA_ON_NOTE)
 
    async def send_picture(label: str | None = None) -> bool:
        """Send one fresh camera picture to Gemini (if the camera is on)."""
        if not getattr(peripherals.camera, "is_on", False):
            return False
        fresh = getattr(peripherals.camera, "_fresh_frame", None)
        if fresh is None:
            return False
        frame = await fresh()
        if frame is None:
            return False
        if label is not None:
            save_debug_frame(frame, label)
        await tell_camera_on()
        await session.send_realtime_input(
            video=types.Blob(data=frame.jpeg, mime_type="image/jpeg")
        )
        return True
 
    async def stream_question_pictures() -> None:
        """While the user is talking, send a picture every second.
 
        Gemini uses pictures that arrive DURING the question. A single
        picture sent right as the user lets go of Talk gets ignored.
        """
        try:
            while True:
                await send_picture()
                await asyncio.sleep(1 / VIDEO_FRAMES_PER_SECOND)
        except asyncio.CancelledError:
            pass
 
    async def start_talking() -> None:
        nonlocal question_streamer, talk_started, told_camera_on
        talk_started = time.monotonic()
        await session.send_realtime_input(activity_start=types.ActivityStart())
        # Tell Gemini right away whether it has a live picture.
        told_camera_on = bool(getattr(peripherals.camera, "is_on", False))
        await session.send_realtime_input(
            text=CAMERA_ON_NOTE if told_camera_on else CAMERA_OFF_NOTE
        )
        # New question: camera back to 1x and retry limits reset.
        state.reset_view()
        state.touch()  # keeps the camera from auto-turning-off mid-use
        new_question = getattr(peripherals.camera, "new_question", None)
        if new_question is not None:
            await new_question()
        # Send pictures while the user asks. Runs even if the camera is off,
        # so turning it on mid-question still works (send_picture skips
        # while it's off).
        question_streamer = asyncio.create_task(stream_question_pictures())
 
    async def end_talking() -> None:
        nonlocal question_streamer
        if question_streamer is not None:
            question_streamer.cancel()
            await asyncio.gather(question_streamer, return_exceptions=True)
            question_streamer = None
        # One last picture of the view when the question ends, then give
        # Gemini a moment to take it in before it starts answering.
        if await send_picture(label="question"):
            print("[Vision] Sent pictures with your question.", flush=True)
            await asyncio.sleep(0.3)
        # Very quick tap = almost no audio in the question. Gemini rejects
        # that (error 1007 "Precondition check failed") and closes the
        # connection, so pad the question with a little silence.
        if time.monotonic() - talk_started < MIN_TALK_S:
            await session.send_realtime_input(
                audio=types.Blob(data=SILENCE_PCM, mime_type="audio/pcm;rate=16000")
            )
        await session.send_realtime_input(activity_end=types.ActivityEnd())
 
    tasks = (
        asyncio.create_task(
            listen_for_capture_button(
                peripherals.talk_button,
                peripherals.audio_input.start_capture,
                peripherals.audio_input.stop_capture,
                lambda: send_audio_frames(peripherals, session),
                status_label="Listening (sending microphone audio)",
                on_capture_started=start_talking,
                on_capture_stopped=end_talking,
                # If the camera button is also held, this is the setup gesture
                # -- don't start talking.
                combo_check=lambda: getattr(peripherals.camera_button, "_pressed", False),
                setup_state=setup_state,
            )
        ),
        asyncio.create_task(
            listen_for_camera_toggle(
                peripherals.camera_button,
                peripherals.camera,
                (lambda: send_video_frames(peripherals, session))
                if STREAM_CAMERA_FRAMES
                else (lambda: keep_camera_running(peripherals)),
                state,
                play_sound=lambda pcm: _play_sound(peripherals, pcm, background),
                # If the talk button is also held, this is the setup gesture
                # -- don't turn the camera on.
                combo_check=lambda: getattr(peripherals.talk_button, "_pressed", False),
                setup_state=setup_state,
            )
        ),
    )
    try:
        done, _ = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            task.result()
        raise RuntimeError("A peripheral button event stream ended unexpectedly.")
    finally:
        if question_streamer is not None:
            question_streamer.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
 
 
async def listen_for_capture_button(
    button: Button,
    start_capture: Callable[[], Awaitable[None]],
    stop_capture: Callable[[], Awaitable[None]],
    send_frames: Callable[[], Awaitable[None]],
    status_label: str,
    on_capture_stopped: Callable[[], Awaitable[None]] | None = None,
    on_capture_started: Callable[[], Awaitable[None]] | None = None,
    combo_check: Callable[[], bool] | None = None,
    setup_state: "SetupState | None" = None,
) -> None:
    """Start streaming while a button is held, then stop on release."""
    events = button.events().__aiter__()
    next_event = asyncio.create_task(anext(events))
    sender: asyncio.Task[None] | None = None
 
    try:
        while True:
            waiting = {next_event}
            if sender is not None:
                waiting.add(sender)
            done, _ = await asyncio.wait(
                waiting,
                return_when=asyncio.FIRST_COMPLETED,
            )
 
            if next_event in done:
                event = next_event.result()
                next_event = asyncio.create_task(anext(events))
 
                if event.pressed and sender is None:
                    # If the other setup button joins within a moment, this is
                    # the Camera+Talk setup gesture -- don't start capturing.
                    if combo_check is not None:
                        await asyncio.sleep(SETUP_COMBO_WINDOW)
                        if combo_check():
                            continue
                    # While setup mode is on, a single button does nothing but
                    # remind the user to leave setup mode first.
                    if setup_state is not None and setup_state.active:
                        await _announce(
                            "Setup mode is on. Please hold camera and talk "
                            "together to leave setup mode first."
                        )
                        continue
                    if on_capture_started is not None:
                        await on_capture_started()
                    await start_capture()
                    sender = asyncio.create_task(send_frames())
                    print(f"[Live] {status_label}.", flush=True)
                elif not event.pressed and sender is not None:
                    await stop_capture()
                    await sender
                    sender = None
                    if on_capture_stopped is not None:
                        await on_capture_stopped()
                    print(f"[Live] Stopped: {status_label}.", flush=True)
 
            # Surface sender failures promptly instead of leaving a failed
            # camera/audio loop hidden while waiting for the next key event.
            if sender is not None and sender in done:
                sender.result()
                raise RuntimeError(
                    "A capture frame stream ended before its button was released."
                )
    finally:
        if not next_event.done():
            next_event.cancel()
        await asyncio.gather(next_event, return_exceptions=True)
 
        if sender is not None:
            await stop_capture()
            if not sender.done():
                await sender
 
 
# ============================================================
# Camera toggle: press C once = on, press again = off
# ============================================================
 
# Camera turns itself off after this long with no questions, so it doesn't
# drain the battery or keep sending pictures if the user forgets it's on.
CAMERA_AUTO_OFF_S = 60.0
 
 
def make_beep(frequency_hz: float, duration_s: float = 0.12, volume: float = 0.25) -> bytes:
    """A short beep as 24 kHz mono 16-bit PCM (same format as Gemini's audio)."""
    rate = AUDIO_OUTPUT_SAMPLE_RATE_HZ
    count = int(rate * duration_s)
    fade = max(1, int(rate * 0.01))  # 10 ms fade in/out so it doesn't click
    samples = array.array("h")
    for i in range(count):
        envelope = min(1.0, i / fade, (count - 1 - i) / fade)
        value = volume * envelope * math.sin(2 * math.pi * frequency_hz * i / rate)
        samples.append(int(value * 32767))
    return samples.tobytes()
 
 
CAMERA_ON_BEEP = make_beep(1200) + make_beep(1600)   # rising = on
CAMERA_OFF_BEEP = make_beep(900) + make_beep(600)    # falling = off
 
 
def _play_sound(peripherals: CompanionPeripherals, pcm: bytes, background: set) -> None:
    """Play a beep without waiting for it to finish."""
    task = asyncio.create_task(peripherals.audio_output.play(AudioOutputFrame(pcm=pcm)))
    background.add(task)
    task.add_done_callback(background.discard)
 
 
async def listen_for_camera_toggle(
    button: Button,
    camera: Any,
    send_frames: Callable[[], Awaitable[None]],
    state: VisionState,
    play_sound: Callable[[bytes], None] | None = None,
    auto_off_s: float = CAMERA_AUTO_OFF_S,
    combo_check: Callable[[], bool] | None = None,
    setup_state: "SetupState | None" = None,
) -> None:
    """Press C: camera on (rising beep). Press C again: off (falling beep).
 
    Also turns off by itself after auto_off_s with no questions.
    """
    events = button.events().__aiter__()
    next_event = asyncio.create_task(anext(events))
    sender: asyncio.Task[None] | None = None
 
    async def turn_on() -> None:
        nonlocal sender
        # The webcam can't be used by vision and recording at once. If a
        # recording is running, say so instead of failing to open the camera.
        if recording.is_recording():
            print("[Live] Camera busy: a recording is using it.", flush=True)
            await _announce("The camera is busy with a recording. Stop the recording to use it.")
            return
        try:
            await camera.start_capture()
        except Exception as error:
            print(f"[Live] Camera couldn't start: {error!r}", flush=True)
            await _announce("The camera isn't available right now.")
            return
        sender = asyncio.create_task(send_frames())
        state.touch()
        if play_sound is not None:
            play_sound(CAMERA_ON_BEEP)
        print("[Live] Camera ON.", flush=True)
 
    async def turn_off(reason: str) -> None:
        nonlocal sender
        await camera.stop_capture()
        if sender is not None:
            try:
                await asyncio.wait_for(sender, timeout=3)
            except Exception as error:  # a stuck/failed sender shouldn't crash IRIS
                print(f"[Live] Camera sender stopped with: {error!r}", flush=True)
            sender = None
        if play_sound is not None:
            play_sound(CAMERA_OFF_BEEP)
        print(f"[Live] Camera OFF ({reason}).", flush=True)
 
    try:
        while True:
            waiting = {next_event}
            timeout = None
            if sender is not None:
                waiting.add(sender)
                timeout = max(0.0, state.last_activity + auto_off_s - time.monotonic())
 
            done, _ = await asyncio.wait(
                waiting, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
 
            # Nothing happened for a while: auto-off.
            if not done:
                if time.monotonic() - state.last_activity >= auto_off_s:
                    await turn_off("no questions for a while")
                continue
 
            if next_event in done:
                event = next_event.result()
                next_event = asyncio.create_task(anext(events))
                # Only the press counts; letting go of C does nothing.
                if event.pressed:
                    # If the talk button joins within a moment, this is the
                    # Camera+Talk setup gesture -- don't toggle the camera.
                    if combo_check is not None:
                        await asyncio.sleep(SETUP_COMBO_WINDOW)
                        if combo_check():
                            continue
                    # While setup mode is on, a single button does nothing but
                    # remind the user to leave setup mode first.
                    if setup_state is not None and setup_state.active:
                        await _announce(
                            "Setup mode is on. Please hold camera and talk "
                            "together to leave setup mode first."
                        )
                        continue
                    if sender is None:
                        await turn_on()
                    else:
                        await turn_off("button")
                continue
 
            # The picture stream stopped on its own (camera problem).
            if sender is not None and sender in done:
                error = None if sender.cancelled() else sender.exception()
                sender = None
                print(f"[Live] Camera stream stopped: {error!r}", flush=True)
                await camera.stop_capture()
                if play_sound is not None:
                    play_sound(CAMERA_OFF_BEEP)
    finally:
        if not next_event.done():
            next_event.cancel()
        await asyncio.gather(next_event, return_exceptions=True)
        if sender is not None:
            await camera.stop_capture()
            if not sender.done():
                sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)