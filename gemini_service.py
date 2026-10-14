"""Function-based Gemini Live API transport skeleton."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from config import Settings
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
        config=types.LiveConnectConfig(response_modalities=["AUDIO"]),
    ) as session:
        await stream_live_session(peripherals, session)


async def stream_live_session(
    peripherals: CompanionPeripherals, session: Any
) -> None:
    """Run response and control tasks until one ends or the session fails."""
    tasks = (
        asyncio.create_task(receive_live_responses(session, peripherals)),
        asyncio.create_task(listen_for_button_events(peripherals, session)),
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
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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
    session: Any, peripherals: CompanionPeripherals
) -> None:
    """Receive model events independently from speaker playback."""
    audio_queue: asyncio.Queue[AudioOutputFrame] = asyncio.Queue()
    tasks = (
        asyncio.create_task(
            receive_live_events(session, peripherals, audio_queue),
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
) -> None:
    """Receive complete model turns continuously and enqueue their audio."""
    speaking = False
    while True:
        # The SDK receive iterator finishes after one complete model turn.
        # Start another iterator so the WebSocket remains ready for the next
        # push-to-talk interaction.
        received_response = False
        async for response in session.receive():
            received_response = True
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


async def listen_for_button_events(
    peripherals: CompanionPeripherals, session: Any
) -> None:
    """Use Space to talk and hold C to share camera frames."""
    tasks = (
        asyncio.create_task(
            listen_for_capture_button(
                peripherals.talk_button,
                peripherals.audio_input.start_capture,
                peripherals.audio_input.stop_capture,
                lambda: send_audio_frames(peripherals, session),
                status_label="Listening (sending microphone audio)",
                on_capture_stopped=lambda: session.send_realtime_input(
                    audio_stream_end=True
                ),
            )
        ),
        asyncio.create_task(
            listen_for_capture_button(
                peripherals.camera_button,
                peripherals.camera.start_capture,
                peripherals.camera.stop_capture,
                lambda: send_video_frames(peripherals, session),
                status_label="Sending camera frames",
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
