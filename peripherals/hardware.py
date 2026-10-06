"""Raspberry Pi 5 peripheral driver skeleton."""

from config import Settings
from interfaces import AudioInput, AudioOutput, Button, Camera, CompanionPeripherals


def create_peripherals(_settings: Settings) -> CompanionPeripherals:
    raise NotImplementedError(
        "Implement and compose the Raspberry Pi 5 audio, camera, and GPIO drivers."
    )


def create_audio_input() -> AudioInput:
    """Create a Pi audio-input driver producing 16 kHz mono PCM frames."""
    raise NotImplementedError("Implement Raspberry Pi microphone capture.")


def create_audio_output() -> AudioOutput:
    """Create a Pi audio-output driver accepting 24 kHz mono PCM frames."""
    raise NotImplementedError("Implement Raspberry Pi speaker playback.")


def create_camera() -> Camera:
    """Create a Pi camera driver producing 640x480 JPEG frames at about 1 FPS."""
    raise NotImplementedError("Implement Raspberry Pi camera capture.")


def create_button() -> Button:
    """Create a GPIO button driver exposing transitions as an async event stream."""
    raise NotImplementedError("Implement asyncio-compatible Raspberry Pi GPIO events.")
