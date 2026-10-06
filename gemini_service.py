"""Function-based Gemini Live API transport skeleton."""

import asyncio

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
    peripherals: CompanionPeripherals, session: object
) -> None:
    """Forward 16 kHz mono PCM microphone frames to the Live session."""
    raise NotImplementedError("Implement microphone-to-Live audio streaming.")


async def send_video_frames(
    peripherals: CompanionPeripherals, session: object
) -> None:
    """Forward 640x480 JPEG camera frames to the Live session."""
    raise NotImplementedError("Implement camera-to-Live video streaming.")


async def receive_live_responses(
    session: object, peripherals: CompanionPeripherals
) -> None:
    """Receive model events and route generated audio to the speaker."""
    raise NotImplementedError("Implement Live response receiving and routing.")


async def handle_live_response(
    response: object, peripherals: CompanionPeripherals
) -> None:
    """Route a single Live API response event to its appropriate output."""
    raise NotImplementedError("Implement handling of a Live API response event.")


async def listen_for_button_events(
    peripherals: CompanionPeripherals, session: object
) -> None:
    """Handle button events such as starting or stopping a conversation."""
    raise NotImplementedError(
        "Implement button-event handling for the active Live session."
    )
