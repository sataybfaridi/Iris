"""Laptop webcam/microphone peripheral driver skeleton."""
import platform
import asyncio
import threading
import logging
import time
import cv2

import sounddevice as sd
from pynput import keyboard

from config import Settings
from interfaces import (
    AUDIO_INPUT_SAMPLE_RATE_HZ,
    AUDIO_OUTPUT_SAMPLE_RATE_HZ,
    VIDEO_FRAME_HEIGHT,
    VIDEO_FRAME_WIDTH,
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

        generation = self._generation
        async with self._lock:
            # Drop frames that were already waiting to play when an interrupt
            # arrived, rather than replaying stale Gemini audio afterward.
            if generation != self._generation:
                return

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
        """Discard pending chunks and stop the speaker after its current write."""
        self._generation += 1
        # play() holds this lock until its blocking write completes. Waiting
        # here avoids racing PortAudio abort against a write, which can cause
        # driver errors and audible clicks/static.
        async with self._lock:
            stream = self._stream
            if stream is None:
                return

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
        """Close the laptop speaker output."""

        async with self._lock:
            stream = self._stream
            if stream is None:
                return

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

# Zoom levels Gemini can use. It can jump straight to any of them.
# Max 3x on the laptop: a 1080p webcam has no real detail past that.
DIGITAL_ZOOM_LEVELS = (1.0, 2.0, 3.0)

# Retry limits per question, so IRIS doesn't keep focusing/zooming forever.
MAX_AUTOFOCUS_RETRIES = 1
MAX_ZOOM_RETRIES = 3

# Ask the webcam for its biggest picture; zooming crops from this, then
# shrinks to VIDEO_FRAME_WIDTH x VIDEO_FRAME_HEIGHT (1280x720).
WEBCAM_REQUEST_SIZE = (1920, 1080)

# After turning on, a webcam needs a moment to fix brightness/focus. The
# first frames are dark, so we wait this long before using any picture.
WEBCAM_WARM_UP_S = 0.4


class LaptopCamera:
    """
    Laptop webcam used for development testing.

    Same features as the Pi camera so the Gemini side can be tested here:
    1280x720 frames, digital zoom around a point, an autofocus retry, and
    the same retry limits and results.

    Zoom here is done in software (cropping the webcam picture). On the
    Pi it's done by the camera itself with ScalerCrop, but the Gemini side
    can't tell the difference.
    """

    def __init__(self) -> None:
        self._capture = None
        self._enabled = False
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)
        self._autofocus_retries = 0
        self._zoom_retries = 0
        # Stops the webcam from closing in the middle of taking a picture
        # (this was the Mac crash when C was tapped quickly).
        self._lock = threading.Lock()

        # A background thread reads the webcam nonstop and keeps only the
        # NEWEST frame. Without this, OpenCV hands back old buffered frames
        # (seconds behind), so Gemini would describe what WAS in front of
        # the camera, not what IS.
        self._latest = None
        self._frame_lock = threading.Lock()
        self._grabber = None
        self._stop_grabbing = threading.Event()

    @property
    def is_on(self) -> bool:
        return self._enabled

    @property
    def zoom_level(self) -> float:
        return self._zoom

    # ------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------

    def _open(self) -> None:
        with self._lock:
            if self._capture is not None:
                return

            for index in (0, 1):
                if platform.system() == "Darwin":
                    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
                else:
                    capture = cv2.VideoCapture(index)

                if not capture.isOpened():
                    capture.release()
                    continue

                capture.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_REQUEST_SIZE[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_REQUEST_SIZE[1])

                success, _ = capture.read()
                if success:
                    self._capture = capture
                    break

                capture.release()

            if self._capture is None:
                raise RuntimeError("Could not open a working laptop camera.")

            with self._frame_lock:
                self._latest = None
            self._stop_grabbing.clear()
            self._grabber = threading.Thread(
                target=self._grab_loop, name="webcam-grabber", daemon=True
            )
            self._grabber.start()

        # Let the webcam warm up so the first picture isn't dark.
        time.sleep(WEBCAM_WARM_UP_S)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._frame_lock:
                if self._latest is not None:
                    return
            time.sleep(0.02)

    def _grab_loop(self) -> None:
        """Keep reading so we always have the newest frame (runs in a thread)."""
        while not self._stop_grabbing.is_set():
            with self._lock:
                capture = self._capture
                if capture is None:
                    return
                success, frame = capture.read()
            if success and frame is not None:
                with self._frame_lock:
                    self._latest = frame
            else:
                time.sleep(0.05)

    def _release(self) -> None:
        self._stop_grabbing.set()
        grabber = self._grabber
        if grabber is not None and grabber is not threading.current_thread():
            grabber.join(timeout=2)
        self._grabber = None
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
        with self._frame_lock:
            self._latest = None

    async def start_capture(self) -> None:
        """Turn the laptop camera on."""
        if self._enabled:
            return
        await asyncio.to_thread(self._open)
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)
        self._enabled = True

    async def stop_capture(self) -> None:
        """Turn the laptop camera off."""
        self._enabled = False
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)
        await asyncio.to_thread(self._release)

    async def close(self) -> None:
        await self.stop_capture()

    # ------------------------------------------------------------
    # Pictures
    # ------------------------------------------------------------

    def _crop(self, frame):
        """Cut the picture to 16:9, then crop for the current zoom."""
        height, width = frame.shape[:2]
        target_ratio = VIDEO_FRAME_WIDTH / VIDEO_FRAME_HEIGHT

        # Most Mac webcams are already 16:9; if not, trim the sides/top.
        if width / height > target_ratio:
            new_width = int(height * target_ratio)
            x0 = (width - new_width) // 2
            frame = frame[:, x0 : x0 + new_width]
        elif width / height < target_ratio:
            new_height = int(width / target_ratio)
            y0 = (height - new_height) // 2
            frame = frame[y0 : y0 + new_height, :]

        if self._zoom <= 1.0:
            return frame

        height, width = frame.shape[:2]
        crop_w = int(width / self._zoom)
        crop_h = int(height / self._zoom)
        cx, cy = self._zoom_center
        x0 = int(cx * width) - crop_w // 2
        y0 = int(cy * height) - crop_h // 2
        x0 = max(0, min(x0, width - crop_w))
        y0 = max(0, min(y0, height - crop_h))
        return frame[y0 : y0 + crop_h, x0 : x0 + crop_w]

    def _read_jpeg(self):
        """JPEG of the NEWEST webcam frame (or None if there isn't one yet)."""
        with self._frame_lock:
            if self._latest is None:
                return None
            frame = self._latest.copy()

        frame = self._crop(frame)
        frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Could not encode camera frame as JPEG.")
        return encoded.tobytes()

    async def frames(self):
        """Yield JPEG frames while the camera is turned on.

        A webcam sometimes fails a read (especially a Mac/iPhone Continuity
        camera). One bad frame shouldn't crash IRIS, so: retry a few times,
        then reopen the webcam, and keep going.
        """
        failures = 0
        while self._enabled:
            jpeg = await asyncio.to_thread(self._read_jpeg)
            if jpeg is not None:
                failures = 0
                yield VideoFrame(jpeg=jpeg)
                continue

            if not self._enabled:
                break

            failures += 1
            if failures < 5:
                await asyncio.sleep(0.1)
                continue

            logger.warning("Laptop camera stopped giving frames, reopening it.")
            print("[Camera] Webcam stopped responding, reopening it.", flush=True)
            failures = 0
            try:
                await asyncio.to_thread(self._release)
                await asyncio.to_thread(self._open)
            except RuntimeError as error:
                logger.warning("Could not reopen laptop camera: %s", error)
                await asyncio.sleep(1.0)

    async def _fresh_frame(self):
        jpeg = await asyncio.to_thread(self._read_jpeg)
        return VideoFrame(jpeg=jpeg) if jpeg is not None else None


    # ------------------------------------------------------------
    # Full-quality pictures for the look tool
    # ------------------------------------------------------------

    def _still(self, zoom: float, center_x: float, center_y: float):
        """Newest frame at FULL webcam resolution, cropped for zoom."""
        with self._frame_lock:
            if self._latest is None:
                return None
            frame = self._latest.copy()

        height, width = frame.shape[:2]
        zoom = max(1.0, float(zoom))
        crop_w, crop_h = int(width / zoom), int(height / zoom)
        x0 = int(center_x * width) - crop_w // 2
        y0 = int(center_y * height) - crop_h // 2
        x0 = max(0, min(x0, width - crop_w))
        y0 = max(0, min(y0, height - crop_h))
        frame = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]

        # Small zoomed crops get enlarged so the model looks at them closely.
        if frame.shape[1] < 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return encoded.tobytes() if success else None

    async def still_jpeg(self, zoom: float = 1.0, center_x: float = 0.5,
                         center_y: float = 0.5):
        if not self._enabled:
            return None
        return await asyncio.to_thread(self._still, zoom, center_x, center_y)

    async def focus_now(self) -> None:
        """Try to refocus (most laptop webcams ignore this), then wait for
        a couple of new frames."""
        await asyncio.to_thread(self._refocus)


    # ------------------------------------------------------------
    # Zoom / focus (same behavior as the Pi camera)
    # ------------------------------------------------------------

    async def new_question(self) -> None:
        """Talk was pressed: back to 1x and fresh retry limits."""
        self._autofocus_retries = 0
        self._zoom_retries = 0
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)

    def _refocus(self) -> bool:
        """Most laptop webcams don't let you trigger focus. Try anyway."""
        with self._lock:
            if self._capture is None:
                return False
            try:
                self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass
        time.sleep(0.3)  # give it a moment
        return True

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
        focused = await asyncio.to_thread(self._refocus)
        frame = await self._fresh_frame()
        print("[Camera] Autofocus retry.", flush=True)
        return self._result(
            "ok" if focused else "focus_failed",
            "Refocused, here is a new picture.",
            frame,
        )

    def _is_same_view(self, level, center_x, center_y, tolerance=0.05) -> bool:
        if level != self._zoom:
            return False
        if level == 1.0:
            return True
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
    """Create a laptop webcam producing 1280x720 JPEG frames at about 1 FPS."""
    return LaptopCamera()

def create_button(key=keyboard.Key.space) -> Button:
    """Create a laptop keyboard button for development testing."""
    return LaptopButton(key)