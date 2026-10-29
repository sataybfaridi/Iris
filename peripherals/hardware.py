"""Raspberry Pi 5 peripheral drivers for hardware deployment mode.

Provisional assumptions (not confirmed final wiring or ALSA device names):
- Audio capture/playback use PortAudio/ALSA defaults (`device=None`).
- Talk button: BCM GPIO17; camera button: BCM GPIO27.
- Buttons are active-low with internal pull-ups and 50 ms debounce.
- Picamera2 is installed via apt (`python3-picamera2`), not requirements.txt.
- Camera works with Camera Module 2 (fixed focus) and Module 3 (autofocus).
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
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


# ============================================================
# CAMERA
# ============================================================

# Zoom levels Gemini can use. It can jump straight to any of them.
# The highest level is picked from the camera's sensor when it turns on
# (see _zoom_levels_for), so we never zoom past real sensor detail much.
ALL_ZOOM_LEVELS = (1.0, 2.0, 3.0, 4.0)

# Retry limits per question, so IRIS doesn't keep focusing/zooming forever.
MAX_AUTOFOCUS_RETRIES = 1
MAX_ZOOM_RETRIES = 3

# After turning on, the camera needs a moment to fix brightness. The first
# frames are dark, so we wait this long before using any picture.
CAMERA_WARM_UP_S = 0.5

# How long to wait for a new zoom to show up in the picture.
SETTLE_TIMEOUT_S = 1.0


def _encode_array_as_jpeg(array) -> bytes:
    """Encode a Picamera2 "RGB888" array as JPEG.

    Careful: Picamera2's "RGB888" is stored as B, G, R in memory. Pillow
    expects R, G, B, so we flip it first; OpenCV already expects B, G, R.
    (Getting this wrong swaps red and blue.)
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        buffer = io.BytesIO()
        Image.fromarray(array[..., ::-1]).save(buffer, format="JPEG")
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

    success, encoded = cv2.imencode(".jpg", array)
    if not success:
        raise RuntimeError("Could not encode camera frame as JPEG.")
    return encoded.tobytes()


def _capture_jpeg_from_picamera(camera) -> bytes:
    """Capture one JPEG from Picamera2 without OpenCV.

    Prefers Picamera2's built-in JPEG writer. Falls back to an array
    encoded with Pillow (or OpenCV only if Pillow is missing).
    """
    buffer = io.BytesIO()
    try:
        camera.capture_file(buffer, format="jpeg")
        jpeg = buffer.getvalue()
        if jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"):
            return jpeg
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        logger.debug("Picamera2 JPEG capture_file unavailable: %s", error)

    return _encode_array_as_jpeg(camera.capture_array())


def _crop_to_tuple(crop):
    """Picamera2 normally gives (x, y, w, h); also accept a Rectangle object."""
    if hasattr(crop, "width"):
        return (int(crop.x), int(crop.y), int(crop.width), int(crop.height))
    x, y, w, h = crop
    return (int(x), int(y), int(w), int(h))


def _crop_close_enough(actual, wanted, tolerance=32) -> bool:
    """The camera can round the crop a little, so allow a small difference."""
    return all(abs(a - w) <= tolerance for a, w in zip(actual, wanted))


def _widescreen_crop(full):
    """Biggest 16:9 area in the middle of the sensor.

    Camera Module 2 and 3 sensors aren't exactly 16:9, but our pictures are
    1280x720. Cropping to 16:9 first means pictures never get stretched.
    """
    x, y, w, h = full
    ratio = VIDEO_FRAME_WIDTH / VIDEO_FRAME_HEIGHT
    if w / h > ratio:
        new_w = int(h * ratio)
        return (x + (w - new_w) // 2, y, new_w, h)
    new_h = int(w / ratio)
    return (x, y + (h - new_h) // 2, w, new_h)


def _zoom_levels_for(crop_width: int):
    """Allow zoom levels that still use real sensor detail (+ a little).

    Module 2 (3280 wide): up to 3x.  Module 3 (4608 wide): up to 4x.
    """
    real_detail = crop_width / VIDEO_FRAME_WIDTH
    levels = tuple(z for z in ALL_ZOOM_LEVELS if z <= real_detail + 0.5)
    return levels or (1.0,)


class PiCamera:
    """Raspberry Pi camera via Picamera2: 1280x720 JPEGs, digital zoom,
    and autofocus retry if the camera has autofocus.

    Same features and results as LaptopCamera in mock.py, so gemini_service
    and vision_tools work the same on both.

    Works with Camera Module 2 (fixed focus, 8MP) and Camera Module 3
    (autofocus, 12MP). It checks which one is plugged in when it turns on.
    """

    def __init__(self) -> None:
        self._camera = None
        self._enabled = False
        self._full_crop = None
        self._zoom_levels = (1.0,)
        self._has_autofocus = False
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)
        self._autofocus_retries = 0
        self._zoom_retries = 0
        # One lock for everything that touches the camera: capture, close,
        # autofocus and zoom can never run at the same time. Without it,
        # pressing the camera button quickly can close the camera in the
        # middle of taking a picture and crash.
        self._lock = threading.Lock()

    @property
    def is_on(self) -> bool:
        return self._enabled

    @property
    def zoom_level(self) -> float:
        return self._zoom

    # ------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------

    def _open_camera(self) -> None:
        """Blocking camera setup. Runs in a worker thread."""
        # Installed via apt (python3-picamera2), not requirements.txt.
        from picamera2 import Picamera2

        with self._lock:
            if self._camera is not None:
                return

            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={
                    "size": (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT),
                    "format": "RGB888",
                },
                # Full sensor, so zooming crops real detail instead of
                # enlarging a small picture. Works for Module 2 and 3.
                sensor={"output_size": camera.sensor_resolution},
            )
            camera.configure(config)
            camera.start()

            # Module 3 has autofocus, Module 2 doesn't (fixed focus).
            self._has_autofocus = "AfMode" in camera.camera_controls
            if self._has_autofocus:
                from libcamera import controls

                camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})

            self._full_crop = _widescreen_crop(
                _crop_to_tuple(camera.camera_properties["ScalerCropMaximum"])
            )
            self._zoom_levels = _zoom_levels_for(self._full_crop[2])
            camera.set_controls({"ScalerCrop": self._full_crop})
            self._camera = camera
            self._zoom = 1.0
            self._zoom_center = (0.5, 0.5)

        # Let the camera fix its brightness so the first picture isn't dark.
        time.sleep(CAMERA_WARM_UP_S)

    def _close_camera(self) -> None:
        with self._lock:
            camera = self._camera
            self._camera = None
            if camera is not None:
                try:
                    camera.stop()
                finally:
                    camera.close()

    async def start_capture(self) -> None:
        """Turn the Pi camera on."""
        if self._enabled:
            return
        # Opening takes about a second; a thread keeps mic/speaker running.
        await asyncio.to_thread(self._open_camera)
        self._enabled = True
        focus = "autofocus" if self._has_autofocus else "fixed focus"
        print(
            f"[Camera] On: {VIDEO_FRAME_WIDTH}x{VIDEO_FRAME_HEIGHT}, {focus}, "
            f"zoom up to {self._zoom_levels[-1]:.0f}x.",
            flush=True,
        )

    async def stop_capture(self) -> None:
        """Turn the Pi camera off."""
        self._enabled = False
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)
        await asyncio.to_thread(self._close_camera)

    async def close(self) -> None:
        await self.stop_capture()

    # ------------------------------------------------------------
    # Pictures
    # ------------------------------------------------------------

    def _capture_locked(self):
        with self._lock:
            if self._camera is None:
                return None
            return _capture_jpeg_from_picamera(self._camera)

    async def frames(self):
        """Yield JPEG frames while the camera is on.

        A capture sometimes fails. One bad frame shouldn't crash IRIS, so
        it waits and tries again. No sleep after a good frame: the caller
        (gemini_service) already waits 1 second between frames.
        """
        while self._enabled:
            try:
                jpeg = await asyncio.to_thread(self._capture_locked)
            except Exception as error:  # one bad frame shouldn't crash IRIS
                logger.warning("Pi camera capture failed: %s", error)
                await asyncio.sleep(0.2)
                continue
            if jpeg is None or not self._enabled:
                break
            yield VideoFrame(jpeg=jpeg)

    async def _fresh_frame(self):
        """One new picture right now (used while the user is asking)."""
        if not self._enabled:
            return None
        try:
            jpeg = await asyncio.to_thread(self._capture_locked)
        except Exception as error:
            logger.warning("Pi camera capture failed: %s", error)
            return None
        return VideoFrame(jpeg=jpeg) if jpeg is not None else None

    # ------------------------------------------------------------
    # Zoom / focus (same behavior as the laptop camera)
    # ------------------------------------------------------------

    async def new_question(self) -> None:
        """Talk was pressed: back to 1x and fresh retry limits."""
        self._autofocus_retries = 0
        self._zoom_retries = 0
        if self._zoom != 1.0:
            await self.zoom_to(1.0, count_as_retry=False)

    def _autofocus_locked(self) -> bool:
        from libcamera import controls

        with self._lock:
            if self._camera is None:
                return False
            self._camera.set_controls({"AfMode": controls.AfModeEnum.Auto})
            try:
                success = self._camera.autofocus_cycle()
            except Exception as error:
                logger.warning("Camera autofocus failed: %s", error)
                success = False
            finally:
                self._camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            return bool(success)

    async def retry_autofocus(self) -> dict:
        """Gemini says it's blurry: focus once, then send a fresh picture."""
        if not self._enabled:
            return self._result("camera_off", "The camera is off.")
        if not self._has_autofocus:
            return self._result(
                "not_supported",
                "This camera has fixed focus and can't refocus. If it's "
                "blurry, ask the user to hold it steady about an arm's "
                "length away.",
            )
        if self._autofocus_retries >= MAX_AUTOFOCUS_RETRIES:
            return self._result(
                "limit_reached",
                "Autofocus was already retried. If it's still blurry, "
                "ask the user to hold the device steady or move closer.",
            )
        self._autofocus_retries += 1
        focused = await asyncio.to_thread(self._autofocus_locked)
        frame = await self._fresh_frame()
        print(f"[Camera] Autofocus retry: {'ok' if focused else 'failed'}.", flush=True)
        return self._result(
            "ok" if focused else "focus_failed",
            "Refocused, here is a new picture." if focused
            else "Autofocus could not lock, here is a new picture anyway.",
            frame,
        )

    def _crop_for(self, zoom: float, center_x: float, center_y: float):
        full_x, full_y, full_width, full_height = self._full_crop
        crop_width = int(full_width / zoom)
        crop_height = int(full_height / zoom)
        crop_x = full_x + int(center_x * full_width) - crop_width // 2
        crop_y = full_y + int(center_y * full_height) - crop_height // 2
        # Keep the crop inside the picture.
        crop_x = max(full_x, min(crop_x, full_x + full_width - crop_width))
        crop_y = max(full_y, min(crop_y, full_y + full_height - crop_height))
        return (crop_x, crop_y, crop_width, crop_height)

    def _set_crop_and_wait_locked(self, crop) -> bool:
        """Set the crop, then wait until frames actually use it."""
        with self._lock:
            if self._camera is None:
                return False
            self._camera.set_controls({"ScalerCrop": crop})
            deadline = time.monotonic() + SETTLE_TIMEOUT_S
            while time.monotonic() < deadline:
                try:
                    metadata = self._camera.capture_metadata()
                except Exception:
                    break
                applied = metadata.get("ScalerCrop")
                if applied is not None and _crop_close_enough(
                    _crop_to_tuple(applied), crop
                ):
                    break
            return True

    def _is_same_view(self, level, center_x, center_y, tolerance=0.05) -> bool:
        if level != self._zoom:
            return False
        if level == 1.0:
            return True  # at 1x the spot doesn't matter, it's the full view
        old_x, old_y = self._zoom_center
        return abs(center_x - old_x) <= tolerance and abs(center_y - old_y) <= tolerance

    async def zoom_to(
        self,
        level: float,
        center_x: float = 0.5,
        center_y: float = 0.5,
        count_as_retry: bool = True,
    ) -> dict:
        """Zoom around a point (0.0-1.0), then send a fresh picture."""
        if not self._enabled or self._full_crop is None:
            return self._result("camera_off", "The camera is off.")

        # Gemini sends these numbers, so keep them in a safe range.
        try:
            level = float(level)
            center_x = float(center_x)
            center_y = float(center_y)
        except (TypeError, ValueError):
            return self._result("bad_request", "Zoom values must be numbers.")

        level = min(self._zoom_levels, key=lambda z: abs(z - level))
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))

        # Same zoom on the same spot = nothing new to see. Don't use up a retry.
        if count_as_retry and self._is_same_view(level, center_x, center_y):
            return self._result(
                "no_change",
                f"Already at {level:.0f}x on that spot. Try a different spot, "
                "or guide the user closer.",
            )

        if count_as_retry and self._zoom_retries >= MAX_ZOOM_RETRIES:
            return self._result(
                "limit_reached",
                "Already zoomed as much as allowed for this question. "
                "Ask the user to move closer to the object.",
            )

        crop = self._crop_for(level, center_x, center_y)
        if not await asyncio.to_thread(self._set_crop_and_wait_locked, crop):
            return self._result("camera_off", "The camera is off.")

        self._zoom = level
        self._zoom_center = (center_x, center_y)
        if count_as_retry:
            self._zoom_retries += 1

        print(f"[Camera] Zoom {level:.0f}x around ({center_x:.2f}, {center_y:.2f}).", flush=True)

        if not count_as_retry:
            return self._result("ok", f"Zoom set to {level:.0f}x.")

        frame = await self._fresh_frame()
        return self._result("ok", f"Zoomed to {level:.0f}x, here is a new picture.", frame)

    async def reset_zoom(self) -> dict:
        return await self.zoom_to(1.0, count_as_retry=False)

    def _result(self, status: str, message: str, frame=None) -> dict:
        """What gemini_service sends back to Gemini after a tool call."""
        zoom_left = max(0, MAX_ZOOM_RETRIES - self._zoom_retries)
        focus_left = max(0, MAX_AUTOFOCUS_RETRIES - self._autofocus_retries)
        return {
            "status": status,
            "message": message,
            "zoom_level": self._zoom,
            "max_zoom": self._zoom_levels[-1],
            "zoom_retries_left": zoom_left,
            "autofocus_retries_left": focus_left if self._has_autofocus else 0,
            "has_autofocus": self._has_autofocus,
            "last_try": zoom_left == 0,
            "ask_user_to_move_closer": status == "limit_reached",
            "frame": frame,
        }


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
        """Audio starts when Talk is held; the camera when its button is pressed."""

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
    """Create a Pi camera driver producing 1280x720 JPEG frames."""
    return PiCamera()


def create_button(pin: int) -> Button:
    """Create a GPIO button driver exposing transitions as an async event stream."""
    return PiButton(pin)