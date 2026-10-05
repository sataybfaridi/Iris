"""Gemini Live API transport skeleton."""

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
    _peripherals: CompanionPeripherals, _session: object
) -> None:
    raise NotImplementedError(
        "Implement bidirectional Live API streaming using the peripheral contracts."
    )
