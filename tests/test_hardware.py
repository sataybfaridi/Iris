import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from config import DeploymentMode, Settings
from interfaces import AudioInputFrame, AudioOutputFrame, ButtonEvent
from peripherals import hardware


class HardwareDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_hardware_factory_uses_provisional_gpio_assignments(self) -> None:
        created_buttons = []

        class FakeGPIOButton:
            def __init__(self, pin, *, pull_up, bounce_time):
                self.pin = pin
                self.pull_up = pull_up
                self.bounce_time = bounce_time
                self.when_pressed = None
                self.when_released = None
                created_buttons.append(self)

            def close(self) -> None:
                pass

        gpiozero = types.ModuleType("gpiozero")
        gpiozero.Button = FakeGPIOButton
        settings = Settings(
            mode=DeploymentMode.HARDWARE,
            gemini_api_key="test-key",
            gemini_live_model="test-model",
        )

        with patch.dict(sys.modules, {"gpiozero": gpiozero}):
            peripherals = hardware.create_peripherals(settings)

        self.assertIsInstance(peripherals.audio_input, hardware.PiAudioInput)
        self.assertIsInstance(peripherals.audio_output, hardware.PiAudioOutput)
        self.assertIsInstance(peripherals.camera, hardware.PiCamera)
        self.assertEqual(
            [device.pin for device in created_buttons],
            [hardware.TALK_BUTTON_GPIO, hardware.CAMERA_BUTTON_GPIO],
        )
        self.assertEqual(
            [device.pull_up for device in created_buttons],
            [True, True],
        )
        with self.assertRaisesRegex(RuntimeError, "GPS is not available"):
            await peripherals.location.current_fix()
        await peripherals.close()

    async def test_gpio_button_publishes_async_press_and_release_events(self) -> None:
        class FakeGPIOButton:
            def __init__(self, *_args, **_kwargs):
                self.when_pressed = None
                self.when_released = None

            def close(self) -> None:
                pass

        gpiozero = types.ModuleType("gpiozero")
        gpiozero.Button = FakeGPIOButton
        with patch.dict(sys.modules, {"gpiozero": gpiozero}):
            button = hardware.PiButton(17)

        async def receive_events():
            events = button.events()
            press_task = asyncio.create_task(anext(events))
            await asyncio.sleep(0)
            button._device.when_pressed()
            pressed = await asyncio.wait_for(press_task, timeout=1)

            release_task = asyncio.create_task(anext(events))
            await asyncio.sleep(0)
            button._device.when_released()
            released = await asyncio.wait_for(release_task, timeout=1)
            await events.aclose()
            return pressed, released

        pressed, released = await receive_events()
        await button.close()

        self.assertEqual(pressed, ButtonEvent(pressed=True, timestamp=pressed.timestamp))
        self.assertEqual(released, ButtonEvent(pressed=False, timestamp=released.timestamp))
        self.assertGreaterEqual(released.timestamp, pressed.timestamp)

    async def test_pi_audio_input_preserves_queued_pcm_when_stopped(self) -> None:
        class FakeInputStream:
            active = False

            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]

            def start(self) -> None:
                self.active = True

            def stop(self) -> None:
                self.callback(b"\x01\x02", 1, None, None)
                self.active = False

            def close(self) -> None:
                pass

        sounddevice = types.ModuleType("sounddevice")
        sounddevice.RawInputStream = FakeInputStream

        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            audio_input = hardware.PiAudioInput()
            await audio_input.start_capture()
            await audio_input.stop_capture()
            frames = [frame async for frame in audio_input.frames()]
            await audio_input.close()

        self.assertEqual(frames, [AudioInputFrame(pcm=b"\x01\x02")])

    async def test_pi_audio_output_uses_24khz_and_interrupts_stream(self) -> None:
        streams = []

        class FakeOutputStream:
            active = False

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.frames = []
                self.stopped = False
                self.closed = False
                streams.append(self)

            def start(self) -> None:
                self.active = True

            def write(self, pcm: bytes) -> None:
                self.frames.append(pcm)

            def stop(self) -> None:
                self.stopped = True
                self.active = False

            def close(self) -> None:
                self.closed = True

        sounddevice = types.ModuleType("sounddevice")
        sounddevice.RawOutputStream = FakeOutputStream
        sounddevice.PortAudioError = OSError

        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            audio_output = hardware.PiAudioOutput()
            await audio_output.play(AudioOutputFrame(pcm=b"\x03\x04"))
            await audio_output.interrupt()

        self.assertEqual(streams[0].kwargs["samplerate"], 24_000)
        self.assertEqual(streams[0].kwargs["channels"], 1)
        self.assertEqual(streams[0].kwargs["device"], None)
        self.assertEqual(streams[0].frames, [b"\x03\x04"])
        self.assertTrue(streams[0].stopped)
        self.assertTrue(streams[0].closed)

    async def test_hardware_module_import_does_not_require_opencv(self) -> None:
        """Importing/creating peripherals must not load OpenCV (Pi BLAS issue)."""
        import builtins
        import importlib

        real_import = builtins.__import__

        def block_cv2(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cv2" or name.startswith("cv2."):
                raise ImportError(
                    "libopenblas.so.0: cannot open shared object file: "
                    "No such file or directory"
                )
            return real_import(name, globals, locals, fromlist, level)

        gpiozero = types.ModuleType("gpiozero")

        class FakeGPIOButton:
            def __init__(self, *_args, **_kwargs):
                self.when_pressed = None
                self.when_released = None

            def close(self) -> None:
                pass

        gpiozero.Button = FakeGPIOButton
        settings = Settings(
            mode=DeploymentMode.HARDWARE,
            gemini_api_key="test-key",
            gemini_live_model="test-model",
        )

        with patch.dict(sys.modules, {"gpiozero": gpiozero, "cv2": None}):
            with patch.object(builtins, "__import__", block_cv2):
                reloaded = importlib.reload(hardware)
                peripherals = reloaded.create_peripherals(settings)
                self.assertIsInstance(peripherals.camera, reloaded.PiCamera)
                await peripherals.close()

        # Restore the module under its normal import path for later tests.
        importlib.reload(hardware)

    async def test_pi_camera_captures_jpeg_without_opencv(self) -> None:
        class FakePicamera2:
            def __init__(self) -> None:
                self.started = False
                self.closed = False
                self.config = None
                self.capture_file_calls = 0

            def create_preview_configuration(self, main):
                self.config = main
                return {"main": main}

            def configure(self, config) -> None:
                self.applied = config

            def start(self) -> None:
                self.started = True

            def capture_file(self, file_output, format=None) -> None:
                self.capture_file_calls += 1
                self.assert_format = format
                file_output.write(b"\xff\xd8picamera-jpeg\xff\xd9")

            def capture_array(self):
                raise AssertionError(
                    "capture_array should not be used when capture_file works"
                )

            def stop(self) -> None:
                self.started = False

            def close(self) -> None:
                self.closed = True

        picamera2 = types.ModuleType("picamera2")
        picamera2.Picamera2 = FakePicamera2

        with patch.dict(sys.modules, {"picamera2": picamera2, "cv2": None}):
            camera = hardware.PiCamera()
            await camera.start_capture()
            fake = camera._camera
            frames = camera.frames()
            frame = await asyncio.wait_for(anext(frames), timeout=2)
            await camera.stop_capture()
            await frames.aclose()
            await camera.close()

        self.assertEqual(fake.capture_file_calls, 1)
        self.assertEqual(fake.assert_format, "jpeg")
        self.assertEqual(frame.jpeg, b"\xff\xd8picamera-jpeg\xff\xd9")

    async def test_pi_camera_falls_back_to_pillow_when_capture_file_unavailable(
        self,
    ) -> None:
        class FakeImage:
            def __init__(self, _array) -> None:
                pass

            def save(self, buffer, format=None) -> None:
                if format != "JPEG":
                    raise AssertionError(f"expected JPEG, got {format!r}")
                buffer.write(b"\xff\xd8pillow-jpeg\xff\xd9")

        class FakePicamera2:
            def __init__(self) -> None:
                self.started = False
                self.closed = False

            def create_preview_configuration(self, main):
                return {"main": main}

            def configure(self, config) -> None:
                pass

            def start(self) -> None:
                self.started = True

            def capture_file(self, file_output, format=None) -> None:
                raise RuntimeError("JPEG still capture not configured")

            def capture_array(self):
                return [[[0, 0, 0]]]

            def stop(self) -> None:
                self.started = False

            def close(self) -> None:
                self.closed = True

        picamera2 = types.ModuleType("picamera2")
        picamera2.Picamera2 = FakePicamera2
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")
        pil_image.fromarray = FakeImage
        pil.Image = pil_image

        with patch.dict(
            sys.modules,
            {
                "picamera2": picamera2,
                "PIL": pil,
                "PIL.Image": pil_image,
                "cv2": None,
            },
        ):
            camera = hardware.PiCamera()
            await camera.start_capture()
            frames = camera.frames()
            frame = await asyncio.wait_for(anext(frames), timeout=2)
            await camera.stop_capture()
            await frames.aclose()
            await camera.close()

        self.assertEqual(frame.jpeg, b"\xff\xd8pillow-jpeg\xff\xd9")
