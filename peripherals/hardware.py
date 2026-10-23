"""Raspberry Pi 5 peripheral drivers for hardware deployment mode.

Provisional assumptions (not confirmed final wiring or ALSA device names):
- Audio capture/playback use PortAudio/ALSA defaults (`device=None`).
- Talk button: BCM GPIO17; camera button: BCM GPIO27.
- Buttons are active-low with internal pull-ups and 50 ms debounce.
- Picamera2 is installed via apt (`python3-picamera2`), not requirements.txt.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Optional

from config import Settings
from interfaces import (
    AUDIO_INPUT_SAMPLE_RATE_HZ,
    AUDIO_OUTPUT_SAMPLE_RATE_HZ,
    VIDEO_FRAME_HEIGHT,
    VIDEO_FRAME_WIDTH,
    VIDEO_FRAMES_PER_SECOND,
    AudioInput,
    AudioInputFrame,
    AudioOutput,
    AudioOutputFrame,
    Button,
    ButtonEvent,
    Camera,
    CompanionPeripherals,
    VideoFrame,
)

logger = logging.getLogger(__name__)

# Provisional GPIO assignments — update to match the device wiring.
TALK_BUTTON_GPIO = 17
CAMERA_BUTTON_GPIO = 27
BUTTON_BOUNCE_TIME_S = 0.05


class PiAudioInput:
    """Wired microphone / ALSA-default capture producing 16 kHz mono PCM."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._stream = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start_capture(self) -> None:
        """Start capturing audio from the ALSA-default input device."""
        import sounddevice as sd

        if self._stream is not None and self._stream.active:
            return

        while not self._queue.empty():
            self._queue.get_nowait()

        self._loop = asyncio.get_running_loop()

        if self._stream is None:
            # device=None: provisional ALSA/PortAudio default capture.
            self._stream = sd.RawInputStream(
                samplerate=AUDIO_INPUT_SAMPLE_RATE_HZ,
                channels=1,
                dtype="int16",
                blocksize=1600,
                device=None,
                callback=self._audio_callback,
            )

        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                bytes(indata),
            )

    async def frames(self):
        """Yield one capture's queued microphone frames until the stop sentinel."""
        while True:
            pcm = await self._queue.get()
            if pcm is None:
                return
            yield AudioInputFrame(pcm=pcm)

    async def stop_capture(self) -> None:
        """Stop capturing and enqueue a sentinel so the sender can drain the queue."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()

        await asyncio.sleep(0)
        self._queue.put_nowait(None)

    async def close(self) -> None:
        """Close the microphone input stream."""
        if self._stream is not None:
            if self._stream.active:
                self._stream.stop()
            self._stream.close()
            self._stream = None


class PiAudioOutput:
    """MAX98357A / ALSA-default speaker accepting 24 kHz mono PCM."""

    def __init__(self) -> None:
        self._stream = None
        self._lock = asyncio.Lock()
        self._generation = 0

    async def play(self, frame: AudioOutputFrame) -> None:
        """Play one audio frame through the ALSA-default output device."""
        import sounddevice as sd

        generation = self._generation
        async with self._lock:
            if generation != self._generation:
                return

            if self._stream is None:
                # device=None: provisional ALSA/PortAudio default playback.
                self._stream = sd.RawOutputStream(
                    samplerate=AUDIO_OUTPUT_SAMPLE_RATE_HZ,
                    channels=1,
                    dtype="int16",
                    device=None,
                )
                self._stream.start()

            stream = self._stream
            try:
                await asyncio.to_thread(stream.write, frame.pcm)
            except sd.PortAudioError:
                if generation != self._generation:
                    return
                raise

    async def interrupt(self) -> None:
        """Discard pending chunks and stop the speaker after its current write."""
        self._generation += 1
        async with self._lock:
            stream = self._stream
            if stream is None:
                return

            import sounddevice as sd

            try:
                try:
                    await asyncio.to_thread(stream.stop)
                except sd.PortAudioError as error:
                    logger.warning(
                        "PortAudio failed to stop interrupted speaker stream: %s",
                        error,
                    )
            finally:
                try:
                    await asyncio.to_thread(stream.close)
                except sd.PortAudioError as error:
                    logger.warning(
                        "PortAudio failed to close interrupted speaker stream: %s",
                        error,
                    )
                finally:
                    self._stream = None

    async def close(self) -> None:
        """Close the speaker output stream."""
        async with self._lock:
            stream = self._stream
            if stream is None:
                return

            import sounddevice as sd

            try:
                try:
                    await asyncio.to_thread(stream.stop)
                except sd.PortAudioError as error:
                    logger.warning("PortAudio failed to stop speaker stream: %s", error)
            finally:
                try:
                    await asyncio.to_thread(stream.close)
                except sd.PortAudioError as error:
                    logger.warning("PortAudio failed to close speaker stream: %s", error)
                finally:
                    self._stream = None


def _encode_rgb_array_as_jpeg(rgb) -> bytes:
    """Encode an RGB array to JPEG without requiring OpenCV at import time.

    Prefers Pillow. Falls back to a lazy OpenCV import only if Pillow is
    unavailable; OpenCV on Raspberry Pi OS often needs ``libopenblas0``.
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG")
        return buffer.getvalue()

    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "JPEG encoding needs Pillow or OpenCV. Install Pillow "
            "(`pip install pillow`) or, if using OpenCV on Raspberry Pi OS, "
            "install the BLAS runtime with `sudo apt install libopenblas0` "
            f"(original error: {error})"
        ) from error

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr)
    if not success:
        raise RuntimeError("Could not encode camera frame as JPEG.")
    return encoded.tobytes()


def _capture_jpeg_from_picamera(camera) -> bytes:
    """Capture one JPEG from Picamera2 without OpenCV.

    Prefers Picamera2's built-in JPEG writer. Falls back to an RGB array
    encoded with Pillow (or lazy OpenCV only if Pillow is missing).
    """
    buffer = io.BytesIO()
    try:
        camera.capture_file(buffer, format="jpeg")
        jpeg = buffer.getvalue()
        if jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"):
            return jpeg
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        logger.debug("Picamera2 JPEG capture_file unavailable: %s", error)

    rgb = camera.capture_array()
    return _encode_rgb_array_as_jpeg(rgb)


class PiCamera:
    """CSI Camera Module 3 Wide via Picamera2, producing 640x480 JPEGs at ~1 FPS."""

    def __init__(self) -> None:
        self._camera = None
        self._enabled = False

    async def start_capture(self) -> None:
        """Turn the Pi camera on (lazy Picamera2 import; no OpenCV)."""
        if self._enabled:
            return

        # Installed via apt (python3-picamera2), not requirements.txt.
        from picamera2 import Picamera2

        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={
                "size": (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT),
                "format": "RGB888",
            }
        )
        camera.configure(config)
        camera.start()
        self._camera = camera
        self._enabled = True

    async def frames(self):
        """Yield JPEG frames while the camera is turned on (~1 FPS)."""
        interval = 1.0 / VIDEO_FRAMES_PER_SECOND

        while self._enabled:
            jpeg = await asyncio.to_thread(_capture_jpeg_from_picamera, self._camera)
            if not self._enabled:
                break

            yield VideoFrame(jpeg=jpeg)
            await asyncio.sleep(interval)

    async def stop_capture(self) -> None:
        """Turn the Pi camera off."""
        self._enabled = False
        camera = self._camera
        self._camera = None
        if camera is not None:
            await asyncio.to_thread(camera.stop)
            await asyncio.to_thread(camera.close)

    async def close(self) -> None:
        """Close the Pi camera."""
        await self.stop_capture()


class PiButton:
    """GPIO push button exposing press/release transitions as an async stream."""

    def __init__(self, pin: int) -> None:
        # Lazy import so tests can inject a fake gpiozero before construction.
        from gpiozero import Button as GPIOButton

        self._pin = pin
        self._queue: asyncio.Queue[ButtonEvent] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # pull_up=True → active-low; bounce_time is seconds (50 ms).
        self._device = GPIOButton(
            pin,
            pull_up=True,
            bounce_time=BUTTON_BOUNCE_TIME_S,
        )
        self._device.when_pressed = self._on_pressed
        self._device.when_released = self._on_released

    def _on_pressed(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                ButtonEvent(pressed=True, timestamp=time.monotonic()),
            )

    def _on_released(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                ButtonEvent(pressed=False, timestamp=time.monotonic()),
            )

    async def events(self):
        """Yield button state transitions from the GPIO callbacks."""
        self._loop = asyncio.get_running_loop()
        while True:
            event = await self._queue.get()
            yield event

    async def close(self) -> None:
        """Release the GPIO button resources."""
        if self._device is not None:
            self._device.close()
            self._device = None


class PiPeripherals:
    """Bundled Pi peripherals; capture stays deferred until matching buttons."""

    def __init__(self) -> None:
        self.audio_input = create_audio_input()
        self.audio_output = create_audio_output()
        self.camera = create_camera()
        self.talk_button = create_button(TALK_BUTTON_GPIO)
        self.camera_button = create_button(CAMERA_BUTTON_GPIO)

    async def start(self) -> None:
        """Defer audio and camera capture until their buttons are held."""

    async def close(self) -> None:
        await self.audio_input.close()
        await self.audio_output.close()
        await self.camera.close()
        await self.talk_button.close()
        await self.camera_button.close()


def create_peripherals(_settings: Settings) -> CompanionPeripherals:
    """Build the Raspberry Pi peripheral bundle used for hardware mode."""
    return PiPeripherals()


def create_audio_input() -> AudioInput:
    """Create a Pi audio-input driver producing 16 kHz mono PCM frames."""
    return PiAudioInput()


def create_audio_output() -> AudioOutput:
    """Create a Pi audio-output driver accepting 24 kHz mono PCM frames."""
    return PiAudioOutput()


def create_camera() -> Camera:
    """Create a Pi camera driver producing 640x480 JPEG frames at about 1 FPS."""
    return PiCamera()


def create_button(pin: int) -> Button:
    """Create a GPIO button driver exposing transitions as an async event stream."""
    return PiButton(pin)
