"""Laptop webcam/microphone peripheral driver skeleton."""
import platform
import asyncio
import time
import cv2

import sounddevice as sd
from pynput import keyboard

from config import Settings
from interfaces import (
    AUDIO_INPUT_SAMPLE_RATE_HZ,
    AUDIO_OUTPUT_SAMPLE_RATE_HZ,
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

class LaptopAudioInput:
    """Laptop microphone driver for development testing."""

    def __init__(self) -> None:
        self._queue = asyncio.Queue()
        self._stream = None
        self._loop = None

    async def start_capture(self) -> None:
        """Start capturing audio from the laptop microphone."""

        if self._stream is not None and self._stream.active:
            return

        while not self._queue.empty():
            self._queue.get_nowait()

        self._loop = asyncio.get_running_loop()

        if self._stream is None:
            self._stream = sd.RawInputStream(
                samplerate=AUDIO_INPUT_SAMPLE_RATE_HZ,
                channels=1,
                dtype="int16",
                blocksize=1600,
                callback=self._audio_callback,
            )

        self._stream.start()

    def _audio_callback(self, indata, frames, time, status) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                bytes(indata),
            )

    async def frames(self):
        """Yield one capture's queued microphone frames."""

        while True:
            pcm = await self._queue.get()
            if pcm is None:
                return
            yield AudioInputFrame(pcm=pcm)

    async def stop_capture(self) -> None:
        """Stop capturing microphone audio."""

        if self._stream is not None and self._stream.active:
            self._stream.stop()

        await asyncio.sleep(0)
        self._queue.put_nowait(None)

    async def close(self) -> None:
        """Close the laptop microphone."""

        if self._stream is not None:
            if self._stream.active:
                self._stream.stop()

            self._stream.close()
            self._stream = None


class LaptopAudioOutput:
    """
    Laptop speaker used for development testing.

    LAPTOP TESTING:
    Plays Gemini's 24 kHz mono PCM audio through the MacBook speaker.

    HARDWARE CHANGE:
    The real IRIS speaker driver will be implemented in
    peripherals/hardware.py.

    Gemini audio will still enter the program as a 24 kHz
    AudioOutputFrame. The hardware driver will handle any
    required conversion, such as 24 kHz to 48 kHz for the
    MAX98357A speaker setup.
    """

    def __init__(self) -> None:
        self._stream = None
        self._lock = asyncio.Lock()
        self._generation = 0

    async def play(self, frame: AudioOutputFrame) -> None:
        """Play one audio frame through the laptop speaker."""

        async with self._lock:
            generation = self._generation
            if self._stream is None:
                self._stream = sd.RawOutputStream(
                    samplerate=AUDIO_OUTPUT_SAMPLE_RATE_HZ,
                    channels=1,
                    dtype="int16",
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
        """Immediately abort queued speaker audio so a user can barge in."""
        self._generation += 1
        stream = self._stream
        if stream is None:
            return

        await asyncio.to_thread(stream.abort)

        async with self._lock:
            if self._stream is stream:
                stream.close()
                self._stream = None

    async def close(self) -> None:
        """Close the laptop speaker output."""

        async with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

class LaptopCamera:
    """
    Laptop webcam used for development testing.

    LAPTOP TESTING:
    Uses the laptop webcam.

    HARDWARE CHANGE:
    The real IRIS device will use the Raspberry Pi Camera Module 3 Wide.
    That camera code will go in peripherals/hardware.py.

    Both versions will provide VideoFrame objects so the Gemini code
    does not need to know which camera is being used.
    """

    def __init__(self) -> None:
        self._capture = None
        self._enabled = False

    async def start_capture(self) -> None:
        """Turn the laptop camera on."""

        if self._enabled:
            return

        for index in (0, 1):
            if platform.system() == "Darwin":
                capture = cv2.VideoCapture(
                    index,
                    cv2.CAP_AVFOUNDATION
                )
            else:
                capture = cv2.VideoCapture(index)

            if not capture.isOpened():
                capture.release()
                continue

            success, _ = capture.read()

            if success:
                self._capture = capture
                break

            capture.release()

        if self._capture is None:
            raise RuntimeError(
                "Could not open a working laptop camera."
            )

        self._capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self._capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        self._enabled = True

    async def frames(self):
        """Yield JPEG frames while the camera is turned on."""

        while self._enabled:
            success, frame = await asyncio.to_thread(
                self._capture.read
            )

            if not success:
                if not self._enabled:
                    break

                raise RuntimeError(
                    "Could not read frame from laptop camera."
                )

            frame = cv2.resize(
                frame,
                (640, 480)
            )

            success, encoded = cv2.imencode(
                ".jpg",
                frame,
            )

            if not success:
                raise RuntimeError(
                    "Could not encode camera frame as JPEG."
                )

            yield VideoFrame(
                jpeg=encoded.tobytes()
            )

    async def stop_capture(self) -> None:
        """Turn the laptop camera off."""

        self._enabled = False

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    async def close(self) -> None:
        """Close the laptop camera."""

        await self.stop_capture()

class LaptopButton:
    """
    Laptop keyboard button used for development testing.

    LAPTOP TESTING:
    Different keyboard keys can represent different IRIS buttons.

    HARDWARE CHANGE:
    The real IRIS device will use physical GPIO buttons in
    peripherals/hardware.py.

    Both versions produce the same ButtonEvent objects.
    """

    def __init__(self, key) -> None:
        self._key = key
        self._queue = asyncio.Queue()
        self._loop = None
        self._listener = None
        self._pressed = False

    def _on_press(self, key) -> None:
        if key == self._key and not self._pressed:
            self._pressed = True

            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait,
                    ButtonEvent(
                        pressed=True,
                        timestamp=time.monotonic(),
                    ),
                )

    def _on_release(self, key) -> None:
        if key == self._key and self._pressed:
            self._pressed = False

            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait,
                    ButtonEvent(
                        pressed=False,
                        timestamp=time.monotonic(),
                    ),
                )

    async def events(self):
        self._loop = asyncio.get_running_loop()

        if self._listener is None:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()

        while True:
            event = await self._queue.get()
            yield event

    async def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

class LaptopPeripherals:
    def __init__(self) -> None:
        self.audio_input = create_audio_input()
        self.audio_output = create_audio_output()
        self.camera = create_camera()

        self.talk_button = create_button(
            keyboard.Key.space
        )

        self.camera_button = create_button(
            keyboard.KeyCode.from_char("c")
        )

    async def start(self) -> None:
        """Defer audio and camera capture until their buttons are held."""

    async def close(self) -> None:
        await self.audio_input.close()
        await self.audio_output.close()
        await self.camera.close()
        await self.talk_button.close()
        await self.camera_button.close()

def create_peripherals(_settings: Settings) -> CompanionPeripherals:
    """Build the laptop peripheral bundle used for mock development."""
    return LaptopPeripherals()


def create_audio_input() -> AudioInput:
    """Create a laptop microphone producing 16 kHz mono PCM frames."""
    return LaptopAudioInput()


def create_audio_output() -> AudioOutput:
    """Create a laptop speaker accepting 24 kHz mono PCM frames."""
    return LaptopAudioOutput()


def create_camera() -> Camera:
    """Create a laptop webcam producing 640x480 JPEG frames at about 1 FPS."""
    return LaptopCamera()

def create_button(key=keyboard.Key.space) -> Button:
    """Create a laptop keyboard button for development testing."""
    return LaptopButton(key)