"""Laptop webcam/microphone peripheral driver skeleton."""

from config import Settings
from interfaces import CompanionPeripherals


def create_peripherals(settings: Settings) -> CompanionPeripherals:
    raise NotImplementedError(
        "Implement laptop webcam and microphone drivers for "
        f"{settings.mode.value} deployment."
    )
