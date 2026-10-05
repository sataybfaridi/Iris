"""Contracts shared by peripheral drivers and the cloud transport."""

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


AUDIO_INPUT_SAMPLE_RATE_HZ = 16_000
AUDIO_OUTPUT_SAMPLE_RATE_HZ = 24_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480
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
    """A complete JPEG image downscaled to 640x480 before delivery."""

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


@runtime_checkable
class AudioInput(Protocol):
    def frames(self) -> AsyncIterator[AudioInputFrame]: ...

    async def close(self) -> None: ...


@runtime_checkable
class AudioOutput(Protocol):
    async def play(self, frame: AudioOutputFrame) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class Camera(Protocol):
    def frames(self) -> AsyncIterator[VideoFrame]: ...

    async def close(self) -> None: ...


@runtime_checkable
class Button(Protocol):
    def events(self) -> AsyncIterator[ButtonEvent]: ...

    async def close(self) -> None: ...


@runtime_checkable
class CompanionPeripherals(Protocol):
    audio_input: AudioInput
    audio_output: AudioOutput
    camera: Camera
    button: Button

    async def start(self) -> None: ...

    async def close(self) -> None: ...
