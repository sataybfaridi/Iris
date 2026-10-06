"""Laptop webcam/microphone peripheral driver skeleton."""

from config import Settings
from interfaces import AudioInput, AudioOutput, Button, Camera, CompanionPeripherals


def create_peripherals(_settings: Settings) -> CompanionPeripherals:
    raise NotImplementedError(
        "Implement and compose the laptop audio, camera, and button drivers."
    )


def create_audio_input() -> AudioInput:
    """Create a laptop microphone producing 16 kHz mono PCM frames."""
    raise NotImplementedError("Implement laptop microphone capture.")


def create_audio_output() -> AudioOutput:
    """Create a laptop speaker accepting 24 kHz mono PCM frames."""
    raise NotImplementedError("Implement laptop speaker playback.")


def create_camera() -> Camera:
    """Create a laptop webcam producing 640x480 JPEG frames at about 1 FPS."""
    raise NotImplementedError("Implement laptop webcam capture.")


def create_button() -> Button:
    """Create a development button-event source for laptop testing."""
    raise NotImplementedError("Implement a mock button-event source.")
