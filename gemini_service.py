"""Function-based Gemini Live API transport skeleton."""

import asyncio
from typing import Any

from config import Settings
from interfaces import CompanionPeripherals


async def run_gemini_live(
    settings: Settings, peripherals: CompanionPeripherals
) -> None:
    """Open a Live session; audio/video queue handling remains to be implemented."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    async with client.aio.live.connect(
        model=settings.gemini_live_model,
        config=types.LiveConnectConfig(response_modalities=["AUDIO"]),
    ) as session:
        await stream_live_session(peripherals, session)


async def stream_live_session(
    peripherals: CompanionPeripherals, session: object
) -> None:
    """Coordinate the microphone, camera, button, and model-response streams."""
    await asyncio.gather(
        send_audio_frames(peripherals, session),
        send_video_frames(peripherals, session),
        receive_live_responses(session, peripherals),
        listen_for_button_events(peripherals, session),
    )


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
    """Forward 640x480 JPEG camera frames to the Live session."""
    from google.genai import types

    async for frame in peripherals.camera.frames():
        video = types.Blob(data=frame.jpeg, mime_type="image/jpeg")
        await session.send_realtime_input(video=video)


async def receive_live_responses(
    session: Any, peripherals: CompanionPeripherals
) -> None:
    """Receive model events and route generated audio to the speaker."""
    # The SDK yields each event as the Live session receives it.
    async for response in session.receive():
        # Keep this loop focused on receiving; response-specific processing
        # belongs in the separate handler below.
        await handle_live_response(response, peripherals)


async def handle_live_response(
    response: object, peripherals: CompanionPeripherals
) -> None:
    """Route a single Live API response event to its appropriate output."""
    


async def listen_for_button_events(
    peripherals: CompanionPeripherals, session: object
) -> None:
    """Handle button events such as starting or stopping a conversation."""
    raise NotImplementedError(
        "Implement button-event handling for the active Live session."
    )
