"""Raspberry Pi 5 peripheral drivers for hardware deployment mode.

Provisional assumptions (not confirmed final wiring or ALSA device names):
- Audio: Google Voice HAT (and similar) run at 48 kHz stereo. Capture is
  mixed to mono and downsampled to 16 kHz for Gemini; playback is upsampled
  from 24 kHz mono to 48 kHz stereo. PortAudio uses ALSA defaults (`device=None`).
- Talk button: BCM GPIO17; camera button: BCM GPIO27.
- Buttons are active-low with internal pull-ups and 50 ms debounce.
- Picamera2 is installed via apt (`python3-picamera2`), not requirements.txt.
- GPS: u-blox NEO-M8N on UART (/dev/serial0, 9600 baud) via NeoM8NLocation.
"""

from __future__ import annotations

import array
import asyncio
import io
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

from config import Settings
from interfaces import (
    AUDIO_INPUT_SAMPLE_RATE_HZ,
    AUDIO_OUTPUT_SAMPLE_RATE_HZ,
    AUDIO_SAMPLE_WIDTH_BYTES,
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
    Location,
    LocationFix,
    VideoFrame,
)
from location.nmea import parse_nmea_line

logger = logging.getLogger(__name__)

# Provisional GPIO assignments — update to match the device wiring.
TALK_BUTTON_GPIO = 17
CAMERA_BUTTON_GPIO = 27
BUTTON_BOUNCE_TIME_S = 0.05
GPS_STALE_AFTER_SECONDS = 15.0
# Google Voice HAT / similar I2S cards: native rate and channel count.
DEVICE_SAMPLE_RATE_HZ = 48_000
DEVICE_CHANNELS = 2
_INPUT_DOWNSAMPLE = DEVICE_SAMPLE_RATE_HZ // AUDIO_INPUT_SAMPLE_RATE_HZ  # 3
_OUTPUT_UPSAMPLE = DEVICE_SAMPLE_RATE_HZ // AUDIO_OUTPUT_SAMPLE_RATE_HZ  # 2
SerialFactory = Callable[[str, int], Any]


def _stereo_i16_to_mono(pcm: bytes) -> bytes:
    """Average L/R int16 samples into mono int16 PCM."""
    if len(pcm) < 4:
        return b""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 4)])
    mono = array.array("h")
    for index in range(0, len(samples) - 1, 2):
        mono.append((samples[index] + samples[index + 1]) // 2)
    return mono.tobytes()


def _downsample_i16_mono(pcm: bytes, factor: int, pending: bytearray) -> bytes:
    """Keep every Nth mono int16 sample; retain incomplete groups in pending."""
    if factor <= 1:
        return pcm
    pending.extend(pcm)
    usable_bytes = len(pending) - (len(pending) % AUDIO_SAMPLE_WIDTH_BYTES)
    if usable_bytes <= 0:
        return b""
    samples = array.array("h")
    samples.frombytes(bytes(pending[:usable_bytes]))
    del pending[:usable_bytes]
    complete = len(samples) - (len(samples) % factor)
    if complete <= 0:
        pending.extend(samples.tobytes())
        return b""
    down = array.array("h", samples[:complete:factor])
    if complete < len(samples):
        pending.extend(samples[complete:].tobytes())
    return down.tobytes()


def _upsample_i16_mono(pcm: bytes, factor: int) -> bytes:
    """Repeat each mono int16 sample `factor` times."""
    if factor <= 1 or not pcm:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % AUDIO_SAMPLE_WIDTH_BYTES)])
    up = array.array("h")
    for sample in samples:
        for _ in range(factor):
            up.append(sample)
    return up.tobytes()


def _mono_i16_to_stereo(pcm: bytes) -> bytes:
    """Duplicate each mono int16 sample into L and R."""
    if not pcm:
        return b""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % AUDIO_SAMPLE_WIDTH_BYTES)])
    stereo = array.array("h")
    for sample in samples:
        stereo.append(sample)
        stereo.append(sample)
    return stereo.tobytes()


class PiAudioInput:
    """Voice HAT capture at 48 kHz stereo, delivered as 16 kHz mono PCM."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._stream = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending = bytearray()

    async def start_capture(self) -> None:
        """Start capturing from the ALSA-default Voice HAT input device."""
        import sounddevice as sd

        if self._stream is not None and self._stream.active:
            return

        while not self._queue.empty():
            self._queue.get_nowait()
        self._pending.clear()

        self._loop = asyncio.get_running_loop()

        if self._stream is None:
            # Native Voice HAT rate; resample to AUDIO_INPUT_SAMPLE_RATE_HZ in callback.
            self._stream = sd.RawInputStream(
                samplerate=DEVICE_SAMPLE_RATE_HZ,
                channels=DEVICE_CHANNELS,
                dtype="int16",
                blocksize=DEVICE_SAMPLE_RATE_HZ // 10,  # ~100 ms
                device=None,
                callback=self._audio_callback,
            )

        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        mono = _stereo_i16_to_mono(bytes(indata))
        pcm16 = _downsample_i16_mono(mono, _INPUT_DOWNSAMPLE, self._pending)
        if not pcm16 or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, pcm16)

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
        self._pending.clear()


class PiAudioOutput:
    """Voice HAT playback: accepts 24 kHz mono, plays as 48 kHz stereo."""

    def __init__(self) -> None:
        self._stream = None
        self._lock = asyncio.Lock()
        self._generation = 0

    async def play(self, frame: AudioOutputFrame) -> None:
        """Play one Gemini audio frame through the ALSA-default output device."""
        import sounddevice as sd

        generation = self._generation
        device_pcm = _mono_i16_to_stereo(
            _upsample_i16_mono(frame.pcm, _OUTPUT_UPSAMPLE)
        )
        async with self._lock:
            if generation != self._generation:
                return

            if self._stream is None:
                self._stream = sd.RawOutputStream(
                    samplerate=DEVICE_SAMPLE_RATE_HZ,
                    channels=DEVICE_CHANNELS,
                    dtype="int16",
                    device=None,
                )
                self._stream.start()

            stream = self._stream
            try:
                await asyncio.to_thread(stream.write, device_pcm)
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


# Match mock camera tool contract (vision/tools.py expects zoom_to / retry_autofocus).
DIGITAL_ZOOM_LEVELS = (1.0, 2.0, 3.0)
MAX_AUTOFOCUS_RETRIES = 1
MAX_ZOOM_RETRIES = 3
GPS_FIX_WAIT_SECONDS = 2.5


class PiCamera:
    """CSI camera via Picamera2 with Live + zoom/focus tool support."""

    def __init__(self) -> None:
        self._camera = None
        self._enabled = False
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)
        self._autofocus_retries = 0
        self._zoom_retries = 0
        self._full_res: tuple[int, int] | None = None

    @property
    def is_on(self) -> bool:
        """True while capture is running. Live uses this for camera notes and pictures."""
        return self._enabled

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
        props = camera.camera_properties or {}
        full = props.get("PixelArraySize") or props.get("ScalerCropMaximum")
        if isinstance(full, (list, tuple)) and len(full) >= 2:
            # ScalerCropMaximum can be (x, y, w, h); PixelArraySize is (w, h).
            if len(full) >= 4:
                self._full_res = (int(full[2]), int(full[3]))
            else:
                self._full_res = (int(full[0]), int(full[1]))
        else:
            self._full_res = (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT)
        self._camera = camera
        self._enabled = True
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)
        self._autofocus_retries = 0
        self._zoom_retries = 0

    async def frames(self):
        """Yield JPEG frames while the camera is turned on (~1 FPS)."""
        interval = 1.0 / VIDEO_FRAMES_PER_SECOND

        while self._enabled:
            jpeg = await asyncio.to_thread(_capture_jpeg_from_picamera, self._camera)
            if not self._enabled:
                break

            yield VideoFrame(jpeg=jpeg)
            await asyncio.sleep(interval)

    async def _fresh_frame(self) -> VideoFrame | None:
        """One JPEG for the current question (Live send_picture / vision tools)."""
        if not self._enabled or self._camera is None:
            return None
        jpeg = await asyncio.to_thread(_capture_jpeg_from_picamera, self._camera)
        return VideoFrame(jpeg=jpeg)

    async def new_question(self) -> None:
        """Talk pressed: reset zoom and retry limits (same as laptop camera)."""
        self._autofocus_retries = 0
        self._zoom_retries = 0
        if self._zoom != 1.0 or self._zoom_center != (0.5, 0.5):
            self._zoom = 1.0
            self._zoom_center = (0.5, 0.5)
            await asyncio.to_thread(self._apply_scaler_crop, 1.0, 0.5, 0.5)

    def _apply_scaler_crop(self, level: float, center_x: float, center_y: float) -> None:
        """Digital zoom via Picamera2 ScalerCrop on the full sensor array."""
        if self._camera is None:
            return
        full_w, full_h = self._full_res or (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT)
        level = max(1.0, float(level))
        crop_w = max(1, int(full_w / level))
        crop_h = max(1, int(full_h / level))
        x0 = int(center_x * full_w) - crop_w // 2
        y0 = int(center_y * full_h) - crop_h // 2
        x0 = max(0, min(x0, full_w - crop_w))
        y0 = max(0, min(y0, full_h - crop_h))
        self._camera.set_controls({"ScalerCrop": (x0, y0, crop_w, crop_h)})

    def _is_same_view(self, level, center_x, center_y, tolerance=0.05) -> bool:
        if level != self._zoom:
            return False
        if level == 1.0:
            return True
        old_x, old_y = self._zoom_center
        return abs(center_x - old_x) <= tolerance and abs(center_y - old_y) <= tolerance

    def _result(self, status: str, message: str, frame=None) -> dict:
        zoom_left = max(0, MAX_ZOOM_RETRIES - self._zoom_retries)
        focus_left = max(0, MAX_AUTOFOCUS_RETRIES - self._autofocus_retries)
        return {
            "status": status,
            "message": message,
            "zoom_level": self._zoom,
            "max_zoom": DIGITAL_ZOOM_LEVELS[-1],
            "zoom_retries_left": zoom_left,
            "autofocus_retries_left": focus_left,
            "last_try": zoom_left == 0,
            "ask_user_to_move_closer": status == "limit_reached",
            "frame": frame,
        }

    async def zoom_to(
        self,
        level: float,
        center_x: float = 0.5,
        center_y: float = 0.5,
        count_as_retry: bool = True,
    ) -> dict:
        """Zoom around a point (0.0-1.0) via ScalerCrop, then send a fresh picture."""
        if not self._enabled:
            return self._result("camera_off", "The camera is off.")

        try:
            level = float(level)
            center_x = float(center_x)
            center_y = float(center_y)
        except (TypeError, ValueError):
            return self._result("bad_request", "Zoom values must be numbers.")

        level = min(DIGITAL_ZOOM_LEVELS, key=lambda z: abs(z - level))
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))

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

        try:
            await asyncio.to_thread(self._apply_scaler_crop, level, center_x, center_y)
        except Exception as error:
            logger.warning("ScalerCrop zoom failed: %s", error)
            return self._result(
                "error",
                "The camera could not zoom. Guide the user closer instead.",
            )

        self._zoom = level
        self._zoom_center = (center_x, center_y)
        if count_as_retry:
            self._zoom_retries += 1

        print(
            f"[Camera] Zoom {level:.0f}x around ({center_x:.2f}, {center_y:.2f}).",
            flush=True,
        )

        if not count_as_retry:
            return self._result("ok", f"Zoom set to {level:.0f}x.")

        # Let the crop apply for a frame before capturing.
        await asyncio.sleep(0.15)
        frame = await self._fresh_frame()
        return self._result(
            "ok", f"Zoomed to {level:.0f}x, here is a new picture.", frame
        )

    def _trigger_autofocus(self) -> bool:
        """Trigger a continuous/auto AF cycle when the module supports it."""
        if self._camera is None:
            return False
        try:
            from libcamera import controls

            self._camera.set_controls({"AfMode": controls.AfModeEnum.Auto})
            cycle = getattr(self._camera, "autofocus_cycle", None)
            if callable(cycle):
                cycle()
            else:
                time.sleep(0.4)
            return True
        except Exception as error:
            # IMX219 (Camera Module 2) is fixed-focus; AF controls are absent.
            logger.info("Autofocus not available on this module: %s", error)
            return False

    async def retry_autofocus(self) -> dict:
        if not self._enabled:
            return self._result("camera_off", "The camera is off.")
        if self._autofocus_retries >= MAX_AUTOFOCUS_RETRIES:
            return self._result(
                "limit_reached",
                "Autofocus was already retried. If it's still blurry, "
                "ask the user to hold the device steady or move closer.",
            )
        self._autofocus_retries += 1
        focused = await asyncio.to_thread(self._trigger_autofocus)
        await asyncio.sleep(0.2)
        frame = await self._fresh_frame()
        print("[Camera] Autofocus retry.", flush=True)
        if focused:
            return self._result(
                "ok", "Refocused, here is a new picture.", frame
            )
        return self._result(
            "focus_failed",
            "This camera could not refocus (it may be fixed-focus). "
            "Here is a new picture; guide the user closer if it's still unclear.",
            frame,
        )

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


def _open_serial_port(port: str, baud_rate: int) -> Any:
    """Open a UART port for NMEA. Imported lazily so laptop tests need no pyserial."""
    import serial

    return serial.Serial(port=port, baudrate=baud_rate, timeout=1.0)


class NeoM8NLocation:
    """u-blox NEO-M8N UART GPS: background NMEA reader, latest valid fix."""

    def __init__(
        self,
        port: str,
        baud_rate: int,
        *,
        serial_factory: SerialFactory | None = None,
        stale_after_seconds: float = GPS_STALE_AFTER_SECONDS,
        fix_wait_seconds: float = GPS_FIX_WAIT_SECONDS,
    ) -> None:
        if not port:
            raise ValueError("GPS serial port must not be empty.")
        if baud_rate <= 0:
            raise ValueError("GPS baud rate must be greater than zero.")
        self._port = port
        self._baud_rate = baud_rate
        self._serial_factory = serial_factory or _open_serial_port
        self._stale_after_seconds = stale_after_seconds
        self._fix_wait_seconds = max(0.0, float(fix_wait_seconds))
        self._lock = threading.Lock()
        self._fix: LocationFix | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial: Any = None
        self._started = False

    def start(self) -> None:
        """Open the UART and begin reading NMEA in a background thread."""
        if self._started:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="neo-m8n-gps",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def _close_serial(self) -> None:
        serial_port = self._serial
        self._serial = None
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass

    def _read_loop(self) -> None:
        """Read NMEA until stopped; reconnect if the UART drops."""
        while not self._stop.is_set():
            try:
                self._serial = self._serial_factory(self._port, self._baud_rate)
            except Exception as error:
                message = str(error).strip() or "GPS is disconnected."
                with self._lock:
                    self._error = "GPS is disconnected."
                logger.warning("GPS serial open failed on %s: %s", self._port, message)
                self._stop.wait(2.0)
                continue

            with self._lock:
                self._error = None
            logger.info("GPS serial open on %s at %s baud", self._port, self._baud_rate)

            try:
                while not self._stop.is_set():
                    try:
                        raw = self._serial.readline()
                    except Exception as error:
                        message = str(error).strip() or "GPS is disconnected."
                        with self._lock:
                            self._error = "GPS is disconnected."
                        logger.warning("GPS serial read failed: %s", message)
                        break
                    if not raw:
                        continue
                    if isinstance(raw, bytes):
                        line = raw.decode("ascii", errors="ignore")
                    else:
                        line = str(raw)
                    fix = parse_nmea_line(line)
                    if fix is None:
                        continue
                    with self._lock:
                        self._fix = fix
                        self._error = None
            finally:
                self._close_serial()

            if not self._stop.is_set():
                self._stop.wait(1.0)

    async def current_fix(self) -> LocationFix:
        """Return the latest valid fix, or raise a speakable GPS error."""
        if not self._started:
            self.start()

        deadline = time.monotonic() + self._fix_wait_seconds
        while True:
            with self._lock:
                fix = self._fix
                error = self._error
            if fix is not None:
                age = time.time() - fix.timestamp
                if age <= self._stale_after_seconds:
                    return fix
            if error and fix is None:
                # Keep waiting a bit in case the reader reconnects mid-question.
                if time.monotonic() >= deadline:
                    raise RuntimeError(error)
            elif time.monotonic() >= deadline:
                raise RuntimeError("Waiting for a GPS fix.")
            await asyncio.sleep(0.05 if self._fix_wait_seconds == 0 else 0.2)

    def apply_nmea_for_tests(self, sentence: str, *, timestamp: float | None = None) -> None:
        """Inject one NMEA sentence without opening a serial port (unit tests)."""
        self._started = True
        fix = parse_nmea_line(sentence, timestamp=timestamp)
        if fix is None:
            return
        with self._lock:
            self._fix = fix
            self._error = None

    async def close(self) -> None:
        """Stop the reader thread and close the serial port."""
        self._stop.set()
        serial_port = self._serial
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 2.0)
        self._started = False
        with self._lock:
            self._fix = None
            self._error = None


class PiPeripherals:
    """Bundled Pi peripherals; capture stays deferred until matching buttons."""

    def __init__(self, settings: Settings) -> None:
        self.audio_input = create_audio_input()
        self.audio_output = create_audio_output()
        self.camera = create_camera()
        self.location = create_location(settings)
        self.talk_button = create_button(TALK_BUTTON_GPIO)
        self.camera_button = create_button(CAMERA_BUTTON_GPIO)

    async def start(self) -> None:
        """Start GPS reading; defer audio and camera until their buttons are held."""
        location = self.location
        start = getattr(location, "start", None)
        if callable(start):
            start()

    async def close(self) -> None:
        await self.audio_input.close()
        await self.audio_output.close()
        await self.camera.close()
        await self.location.close()
        await self.talk_button.close()
        await self.camera_button.close()


def create_peripherals(settings: Settings) -> CompanionPeripherals:
    """Build the Raspberry Pi peripheral bundle used for hardware mode."""
    return PiPeripherals(settings)


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


def create_location(settings: Settings) -> Location:
    """Create a NEO-M8N UART location driver from settings."""
    return NeoM8NLocation(
        port=settings.gps_serial_port,
        baud_rate=settings.gps_baud_rate,
    )
