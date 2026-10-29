"""Contracts shared by peripheral drivers and the cloud transport."""

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


AUDIO_INPUT_SAMPLE_RATE_HZ = 16_000
AUDIO_OUTPUT_SAMPLE_RATE_HZ = 24_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
VIDEO_FRAME_WIDTH = 1280
VIDEO_FRAME_HEIGHT = 720
VIDEO_FRAMES_PER_SECOND = 1



def _validate_pcm(data: bytes) -> None:
    if not isinstance(data, bytes):
        raise TypeError("PCM payload must be bytes.")
    if not data:
        raise ValueError("PCM payload must contain at least one sample.")
    if len(data) % AUDIO_SAMPLE_WIDTH_BYTES:
        raise ValueError("16-bit PCM payload length must be an even number of bytes.")


@dataclass(frozen=True)
class AudioInputFrame:
    """Mono, signed 16-bit little-endian PCM sampled at 16 kHz."""

    pcm: bytes

    def __post_init__(self) -> None:
        _validate_pcm(self.pcm)


@dataclass(frozen=True)
class AudioOutputFrame:
    """Mono, signed 16-bit little-endian PCM sampled at 24 kHz."""

    pcm: bytes

    def __post_init__(self) -> None:
        _validate_pcm(self.pcm)


@dataclass(frozen=True)
class VideoFrame:
    """A complete JPEG image downscaled to 1280x720 before delivery."""

    jpeg: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.jpeg, bytes):
            raise TypeError("JPEG payload must be bytes.")
        if not self.jpeg.startswith(b"\xff\xd8") or not self.jpeg.endswith(b"\xff\xd9"):
            raise ValueError("Video frame must contain a complete JPEG image.")


@dataclass(frozen=True)
class ButtonEvent:
    """A button state transition delivered from an async event stream."""

    pressed: bool
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.pressed, bool):
            raise TypeError("pressed must be a bool.")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative.")


def _validate_coordinate(name: str, value: float, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}.")


@dataclass(frozen=True)
class LocationFix:
    """A GPS fix: WGS84 degrees, accuracy in meters, and a Unix timestamp."""

    latitude: float
    longitude: float
    accuracy_meters: float
    timestamp: float

    def __post_init__(self) -> None:
        _validate_coordinate("latitude", self.latitude, -90, 90)
        _validate_coordinate("longitude", self.longitude, -180, 180)
        if isinstance(self.accuracy_meters, bool) or not isinstance(
            self.accuracy_meters, (int, float)
        ):
            raise TypeError("accuracy_meters must be a number.")
        if self.accuracy_meters < 0:
            raise ValueError("accuracy_meters must be zero or greater.")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be a number.")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative.")


@runtime_checkable
class AudioInput(Protocol):
    async def start_capture(self) -> None: ...

    def frames(self) -> AsyncIterator[AudioInputFrame]: ...

    async def stop_capture(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class AudioOutput(Protocol):
    async def play(self, frame: AudioOutputFrame) -> None: ...

    async def interrupt(self) -> None: ...

    async def close(self) -> None: ...

@runtime_checkable
class Camera(Protocol):
    async def start_capture(self) -> None: ...

    def frames(self) -> AsyncIterator[VideoFrame]: ...

    async def stop_capture(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class Button(Protocol):
    def events(self) -> AsyncIterator[ButtonEvent]: ...

    async def close(self) -> None: ...


@runtime_checkable
class Location(Protocol):
    async def current_fix(self) -> LocationFix: ...

    async def close(self) -> None: ...


@runtime_checkable
class CompanionPeripherals(Protocol):
    audio_input: AudioInput
    audio_output: AudioOutput
    camera: Camera
    talk_button: Button
    camera_button: Button
    location: Location

    async def start(self) -> None: ...

    async def close(self) -> None: ...