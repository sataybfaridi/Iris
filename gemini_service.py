"""Function-based Gemini Live API transport skeleton."""

import array
import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any

from config import DeploymentMode, Settings
from location_tools import (
    LOCATION_TOOL_NAMES,
    build_location_tools,
    handle_location_tool_call,
)
from navigation import NavigationController
from vision_tools import (
    FOCUS_TOOL,
    VisionState,
    ZOOM_TOOL,
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


async def run_gemini_live(
    settings: Settings, peripherals: CompanionPeripherals
) -> None:
    """Open the Live session and run its concurrent interaction tasks."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    async with client.aio.live.connect(
        model=settings.gemini_live_model,
        config=types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            # Let Gemini see pictures at full quality (small label text).
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            # Camera tools + rules. The laptop webcam faces you, so left and
            # right are flipped; the Pi camera points away like a phone.
            system_instruction=build_system_instruction(
                camera_faces_user=settings.mode is DeploymentMode.MOCK
            ),
            tools=[build_camera_tools(types), build_location_tools(types)],
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        ),
    ) as session:
        await stream_live_session(
            peripherals, session, settings.google_maps_api_key
        )


async def stream_live_session(
    peripherals: CompanionPeripherals,
    session: Any,
    maps_api_key: str = "",
) -> None:
    """Run response and control tasks until one ends or the session fails."""
    state = VisionState()
    user_idle = asyncio.Event()
    user_idle.set()
    navigator = NavigationController(
        lambda phrase: speak_navigation_cue(session, phrase),
        idle=user_idle,
    )
    tasks = (
        asyncio.create_task(
            receive_live_responses(
                session, peripherals, state, maps_api_key, navigator
            )
        ),
        asyncio.create_task(
            listen_for_button_events(peripherals, session, state, user_idle)
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
            "Gemini Live interaction task(s) ended unexpectedly: "
            f"{finished_names}."
        )
    finally:
        navigator.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def speak_navigation_cue(session: Any, phrase: str) -> None:
    """Ask Gemini to say one GPS sentence and nothing else."""
    from google.genai import types

    print(f"[Navigation] {phrase}", flush=True)
    await session.send_realtime_input(activity_start=types.ActivityStart())
    await session.send_realtime_input(text=f"[Navigation cue: {phrase}]")
    await session.send_realtime_input(activity_end=types.ActivityEnd())


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
    maps_api_key: str = "",
    navigator: NavigationController | None = None,
) -> None:
    """Receive model events independently from speaker playback."""
    audio_queue: asyncio.Queue[AudioOutputFrame] = asyncio.Queue()
    tasks = (
        asyncio.create_task(
            receive_live_events(
                session, peripherals, audio_queue, state, maps_api_key, navigator
            ),
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
    maps_api_key: str = "",
    navigator: NavigationController | None = None,
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

            # Gemini asked for a camera, location, or directions tool.
            # Run it in the background so we keep receiving.
            tool_call = getattr(response, "tool_call", None)
            if tool_call is not None:
                task = asyncio.create_task(
                    dispatch_tool_call(
                        tool_call,
                        session,
                        peripherals,
                        state,
                        maps_api_key,
                        navigator,
                    )
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


class _ToolCalls:
    def __init__(self, function_calls: list) -> None:
        self.function_calls = function_calls


def _split_tool_calls(tool_call: Any) -> tuple[_ToolCalls | None, _ToolCalls | None, _ToolCalls | None]:
    camera_names = {ZOOM_TOOL, FOCUS_TOOL}
    camera, location, unknown = [], [], []
    for call in tool_call.function_calls or ():
        name = getattr(call, "name", None)
        if name in camera_names:
            camera.append(call)
        elif name in LOCATION_TOOL_NAMES:
            location.append(call)
        else:
            unknown.append(call)
    return (
        _ToolCalls(camera) if camera else None,
        _ToolCalls(location) if location else None,
        _ToolCalls(unknown) if unknown else None,
    )


async def dispatch_tool_call(
    tool_call: Any,
    session: Any,
    peripherals: CompanionPeripherals,
    state: VisionState,
    maps_api_key: str,
    navigator: NavigationController | None = None,
) -> None:
    """Send camera calls to the camera handler and location calls to Maps."""
    camera_call, location_call, unknown_call = _split_tool_calls(tool_call)
    if camera_call is not None:
        await handle_tool_call(camera_call, session, peripherals.camera, state)
    if unknown_call is not None:
        await handle_tool_call(unknown_call, session, peripherals.camera, state)
    if location_call is not None:
        await handle_location_tool_call(
            location_call,
            session,
            peripherals.location,
            maps_api_key,
            navigator,
        )


def _finish_tool_task(tool_tasks: set):
    def done(task: asyncio.Task) -> None:
        tool_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            print(f"[Live] Tool call failed: {task.exception()}", flush=True)

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
    user_idle: asyncio.Event | None = None,
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
        if user_idle is not None:
            user_idle.clear()
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
        if user_idle is not None:
            user_idle.set()

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
) -> None:
    """Press C: camera on (rising beep). Press C again: off (falling beep).

    Also turns off by itself after auto_off_s with no questions.
    """
    events = button.events().__aiter__()
    next_event = asyncio.create_task(anext(events))
    sender: asyncio.Task[None] | None = None

    async def turn_on() -> None:
        nonlocal sender
        await camera.start_capture()
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