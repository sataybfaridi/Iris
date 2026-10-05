"""Raspberry Pi 5 peripheral driver skeleton."""

from config import Settings
from interfaces import CompanionPeripherals


def create_peripherals(settings: Settings) -> CompanionPeripherals:
    raise NotImplementedError(
        "Implement Raspberry Pi 5 audio, camera, and asyncio button drivers "
        f"for {settings.mode.value} deployment."
    )
