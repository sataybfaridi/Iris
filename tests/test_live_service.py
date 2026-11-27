import asyncio
from contextlib import redirect_stdout
from io import StringIO
import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_stub)

import config
from gemini import service as gemini_service
from interfaces import AudioInputFrame, AudioOutputFrame, LocationFix
from location import directions, tools as location_tools


class FakeBlob:
    def __init__(self, *, data: bytes, mime_type: str) -> None:
        self.data = data
        self.mime_type = mime_type


class FakeAudioInput:
    async def frames(self):
        yield AudioInputFrame(pcm=b"\x01\x02")


class FakeCamera:
    async def frames(self):
        yield SimpleNamespace(jpeg=b"\xff\xd8jpeg\xff\xd9")


class FakeSession:
    def __init__(self) -> None:
        self.inputs = []

    async def send_realtime_input(self, **kwargs) -> None:
        self.inputs.append(kwargs)


class LiveServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_audio_frames_sends_pcm_at_input_rate(self) -> None:
        google_module = types.ModuleType("google")
        google_module.__path__ = []
        genai_module = types.ModuleType("google.genai")
        genai_module.types = SimpleNamespace(Blob=FakeBlob)
        google_module.genai = genai_module
        peripherals = SimpleNamespace(audio_input=FakeAudioInput())
        session = FakeSession()

        with patch.dict(
            sys.modules,
            {"google": google_module, "google.genai": genai_module},
        ):
            await gemini_service.send_audio_frames(peripherals, session)

        audio = session.inputs[0]["audio"]
        self.assertEqual(audio.data, b"\x01\x02")
        self.assertEqual(audio.mime_type, "audio/pcm;rate=16000")

    async def test_send_video_frames_sends_jpeg_at_most_once_per_second(self) -> None:
        google_module = types.ModuleType("google")
        google_module.__path__ = []
        genai_module = types.ModuleType("google.genai")
        genai_module.types = SimpleNamespace(Blob=FakeBlob)
        google_module.genai = genai_module
        peripherals = SimpleNamespace(camera=FakeCamera())
        session = FakeSession()

        with (
            patch.dict(
                sys.modules,
                {"google": google_module, "google.genai": genai_module},
            ),
            patch.object(gemini_service.asyncio, "sleep", new=AsyncMock()) as sleep,
        ):
            await gemini_service.send_video_frames(peripherals, session)

        video = session.inputs[0]["video"]
        self.assertEqual(video.data, b"\xff\xd8jpeg\xff\xd9")
        self.assertEqual(video.mime_type, "image/jpeg")
        sleep.assert_awaited_once_with(1.0)

    async def test_handle_live_response_plays_pcm_audio_chunk(self) -> None:
        played = []

        class FakeAudioOutput:
            async def play(self, frame) -> None:
                played.append(frame)

            async def interrupt(self) -> None:
                pass

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                interrupted=False,
                turn_complete=True,
                model_turn=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(
                                mime_type="audio/pcm;rate=24000",
                                data=b"\x03\x04",
                            )
                        )
                    ]
                )
            )
        )
        peripherals = SimpleNamespace(audio_output=FakeAudioOutput())
        audio_queue = asyncio.Queue()

        with redirect_stdout(StringIO()) as output:
            speaking = await gemini_service.handle_live_response(
                response, peripherals, audio_queue
            )

        self.assertEqual(
            audio_queue.get_nowait(),
            AudioOutputFrame(pcm=b"\x03\x04"),
        )
        self.assertFalse(played)
        self.assertFalse(speaking)
        self.assertIn("Speaking", output.getvalue())
        self.assertIn("finished speaking", output.getvalue())

    async def test_interrupted_event_clears_queue_and_interrupts_speaker(self) -> None:
        audio_queue = asyncio.Queue()
        audio_queue.put_nowait(AudioOutputFrame(pcm=b"\x01\x02"))
        speaker_interrupts = []

        class FakeAudioOutput:
            async def play(self, frame) -> None:
                raise AssertionError("Interrupted audio must not be played.")

            async def interrupt(self) -> None:
                speaker_interrupts.append(True)

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                interrupted=True,
                model_turn=None,
                turn_complete=False,
            )
        )

        with redirect_stdout(StringIO()) as output:
            speaking = await gemini_service.handle_live_response(
                response,
                SimpleNamespace(audio_output=FakeAudioOutput()),
                audio_queue,
                speaking=True,
            )

        self.assertTrue(audio_queue.empty())
        self.assertEqual(speaker_interrupts, [True])
        self.assertFalse(speaking)
        self.assertIn("Interrupted", output.getvalue())

    async def test_live_receiver_does_not_wait_for_speaker_playback(self) -> None:
        audio_queue = asyncio.Queue()
        playback_started = asyncio.Event()
        unblock_playback = asyncio.Event()
        speaker_interrupts = []

        class FakeAudioOutput:
            async def play(self, frame) -> None:
                playback_started.set()
                await unblock_playback.wait()

            async def interrupt(self) -> None:
                speaker_interrupts.append(True)
                unblock_playback.set()

        response_with_audio = SimpleNamespace(
            server_content=SimpleNamespace(
                interrupted=False,
                model_turn=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(
                                mime_type="audio/pcm;rate=24000",
                                data=b"\x01\x02",
                            )
                        )
                    ]
                ),
                turn_complete=False,
            )
        )
        interruption_response = SimpleNamespace(
            server_content=SimpleNamespace(
                interrupted=True,
                model_turn=None,
                turn_complete=False,
            )
        )
        received = 0

        class FakeSession:
            async def receive(self):
                nonlocal received
                received += 1
                if received == 1:
                    yield response_with_audio
                    return
                if received == 2:
                    await playback_started.wait()
                    yield interruption_response
                    await asyncio.Event().wait()
                    return
                raise AssertionError("Unexpected receive iteration.")

        peripherals = SimpleNamespace(audio_output=FakeAudioOutput())
        receiver = asyncio.create_task(
            gemini_service.receive_live_responses(FakeSession(), peripherals)
        )
        try:
            while not speaker_interrupts:
                await asyncio.sleep(0)
            self.assertEqual(received, 2)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    async def test_laptop_audio_output_interrupt_waits_for_current_write(self) -> None:
        from peripherals.mock import LaptopAudioOutput

        write_started = threading.Event()
        allow_write_to_finish = threading.Event()
        stream_closed = []

        class BlockingOutputStream:
            active = True

            def start(self) -> None:
                pass

            def write(self, pcm: bytes) -> None:
                write_started.set()
                allow_write_to_finish.wait(timeout=2)

            def stop(self) -> None:
                stream_stopped.append(True)

            def close(self) -> None:
                stream_closed.append(True)

        output_stream = BlockingOutputStream()
        stream_stopped = []
        output = LaptopAudioOutput()
        with patch("peripherals.mock.sd.RawOutputStream", return_value=output_stream):
            play_task = asyncio.create_task(
                output.play(AudioOutputFrame(pcm=b"\x01\x02"))
            )
            self.assertTrue(
                await asyncio.to_thread(write_started.wait, 1),
                "Expected the speaker write to start.",
            )

            interrupt_task = asyncio.create_task(output.interrupt())
            await asyncio.sleep(0.01)
            self.assertFalse(interrupt_task.done())
            self.assertFalse(stream_stopped)

            allow_write_to_finish.set()
            await play_task
            await interrupt_task

        self.assertEqual(stream_stopped, [True])
        self.assertEqual(stream_closed, [True])
        self.assertIsNone(output._stream)

    async def test_laptop_audio_output_recovers_from_interrupt_portaudio_errors(self) -> None:
        from peripherals.mock import LaptopAudioOutput
        import sounddevice as sd

        class FailingOutputStream:
            active = True

            def start(self) -> None:
                pass

            def write(self, pcm: bytes) -> None:
                pass

            def stop(self) -> None:
                raise sd.PortAudioError("stream already stopping")

            def close(self) -> None:
                raise sd.PortAudioError("stream already closed")

        output = LaptopAudioOutput()
        with patch(
            "peripherals.mock.sd.RawOutputStream",
            return_value=FailingOutputStream(),
        ):
            await output.play(AudioOutputFrame(pcm=b"\x01\x02"))
            with self.assertLogs("peripherals.mock", level="WARNING") as logs:
                await output.interrupt()

        self.assertIsNone(output._stream)
        self.assertEqual(len(logs.records), 2)
        self.assertIn("failed to stop interrupted", logs.records[0].getMessage())
        self.assertIn("failed to close", logs.records[1].getMessage())

    async def test_laptop_audio_output_close_recovers_from_portaudio_errors(self) -> None:
        from peripherals.mock import LaptopAudioOutput
        import sounddevice as sd

        class FailingOutputStream:
            def stop(self) -> None:
                raise sd.PortAudioError("stream already stopped")

            def close(self) -> None:
                raise sd.PortAudioError("stream already closed")

        output = LaptopAudioOutput()
        output._stream = FailingOutputStream()

        with self.assertLogs("peripherals.mock", level="WARNING") as logs:
            await output.close()

        self.assertIsNone(output._stream)
        self.assertEqual(len(logs.records), 2)
        self.assertIn("failed to stop", logs.records[0].getMessage())
        self.assertIn("failed to close", logs.records[1].getMessage())

    async def test_talk_button_starts_and_stops_capture(self) -> None:
        states = []
        active = False
        audio_ended = []

        class FakeButton:
            async def events(self):
                yield SimpleNamespace(pressed=True)
                yield SimpleNamespace(pressed=False)
                await asyncio.Event().wait()

        async def start_capture() -> None:
            nonlocal active
            active = True
            states.append("start")

        async def stop_capture() -> None:
            nonlocal active
            active = False
            states.append("stop")

        async def send_frames() -> None:
            while active:
                await asyncio.sleep(0)

        async def end_audio() -> None:
            audio_ended.append(True)

        listener = asyncio.create_task(
            gemini_service.listen_for_capture_button(
                FakeButton(),
                start_capture,
                stop_capture,
                send_frames,
                status_label="Listening",
                on_capture_stopped=end_audio,
            )
        )
        while not audio_ended:
            await asyncio.sleep(0)
        listener.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await listener

        self.assertEqual(states, ["start", "stop"])
        self.assertEqual(audio_ended, [True])

    async def test_receive_live_responses_reopens_iterator_after_turn(self) -> None:
        received_turns = 0
        handled_responses = []
        end_of_session = ConnectionError("Live connection ended")

        class FakeSession:
            async def receive(self):
                nonlocal received_turns
                received_turns += 1
                if received_turns > 2:
                    raise end_of_session
                yield SimpleNamespace(server_content=None)

        async def handle(response, peripherals, audio_queue, speaking=False):
            handled_responses.append(response)
            return speaking

        with patch.object(gemini_service, "handle_live_response", new=handle):
            with self.assertRaisesRegex(ConnectionError, "Live connection ended"):
                await gemini_service.receive_live_events(
                    FakeSession(), object(), asyncio.Queue()
                )

        self.assertEqual(received_turns, 3)
        self.assertEqual(len(handled_responses), 2)

    def test_settings_use_model_environment_value(self) -> None:
        environment = {
            "DEPLOYMENT_MODE": "mock",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_LIVE_MODEL": "test-live-model",
            "MOCK_LATITUDE": "37.7",
            "MOCK_LONGITUDE": "-122.4",
            "MOCK_LOCATION_ACCURACY_METERS": "8",
            "GOOGLE_MAPS_API_KEY": "maps-key",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = config.Settings.from_env()

        self.assertEqual(settings.gemini_live_model, "test-live-model")
        self.assertEqual(settings.mock_latitude, 37.7)
        self.assertEqual(settings.mock_longitude, -122.4)
        self.assertEqual(settings.mock_location_accuracy_meters, 8)
        self.assertEqual(settings.google_maps_api_key, "maps-key")

    def test_settings_default_to_live_api_model(self) -> None:
        environment = {
            "DEPLOYMENT_MODE": "mock",
            "GEMINI_API_KEY": "test-key",
            "MOCK_LATITUDE": "37.7",
            "MOCK_LONGITUDE": "-122.4",
            "MOCK_LOCATION_ACCURACY_METERS": "8",
            "GOOGLE_MAPS_API_KEY": "maps-key",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = config.Settings.from_env()

        self.assertEqual(settings.gemini_live_model, "gemini-3.8-live")

    def test_mock_mode_requires_location_and_maps_settings(self) -> None:
        environment = {
            "DEPLOYMENT_MODE": "mock",
            "GEMINI_API_KEY": "test-key",
            # Explicit empties so load_dotenv() cannot refill from a local .env.
            "MOCK_LATITUDE": "",
            "MOCK_LONGITUDE": "",
            "MOCK_LOCATION_ACCURACY_METERS": "",
            "GOOGLE_MAPS_API_KEY": "",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ValueError):
                config.Settings.from_env()

    async def test_location_tool_response_matches_the_fix(self) -> None:
        class FakeFunctionResponse:
            def __init__(self, *, id, name, response) -> None:
                self.id = id
                self.name = name
                self.response = response

        class Session:
            def __init__(self) -> None:
                self.responses = []

            async def send_tool_response(self, function_responses) -> None:
                self.responses.extend(function_responses)

        class Location:
            async def current_fix(self):
                return LocationFix(37.7, -122.4, 8, timestamp=1)

        google_module = types.ModuleType("google")
        google_module.__path__ = []
        genai_module = types.ModuleType("google.genai")
        genai_module.types = SimpleNamespace(FunctionResponse=FakeFunctionResponse)
        session = Session()
        tool_call = SimpleNamespace(
            function_calls=[
                SimpleNamespace(id="loc-1", name="get_current_location", args={})
            ]
        )

        with patch.dict(
            sys.modules,
            {"google": google_module, "google.genai": genai_module},
        ):
            await location_tools.handle_location_tool_call(
                tool_call, session, Location(), "maps-key"
            )

        self.assertEqual(len(session.responses), 1)
        self.assertEqual(session.responses[0].name, "get_current_location")
        self.assertEqual(
            session.responses[0].response,
            {
                "status": "ok",
                "latitude": 37.7,
                "longitude": -122.4,
                "accuracy_meters": 8,
            },
        )

    async def test_directions_tool_response_speaks_parsed_steps(self) -> None:
        class FakeFunctionResponse:
            def __init__(self, *, id, name, response) -> None:
                self.id = id
                self.name = name
                self.response = response

        class Session:
            def __init__(self) -> None:
                self.responses = []

            async def send_tool_response(self, function_responses) -> None:
                self.responses.extend(function_responses)

        class Location:
            async def current_fix(self):
                return LocationFix(37.7, -122.4, 8, timestamp=1)

        google_module = types.ModuleType("google")
        google_module.__path__ = []
        genai_module = types.ModuleType("google.genai")
        genai_module.types = SimpleNamespace(FunctionResponse=FakeFunctionResponse)
        session = Session()
        tool_call = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    id="dir-1",
                    name="get_directions",
                    args={"destination": "the pharmacy"},
                )
            ]
        )
        fixture = {
            "routes": [
                {
                    "legs": [
                        {
                            "distanceMeters": 500,
                            "duration": "360s",
                            "steps": [
                                {
                                    "distanceMeters": 500,
                                    "staticDuration": "360s",
                                    "navigationInstruction": {
                                        "instructions": "Head north"
                                    },
                                }
                            ],
                        }
                    ]
                }
            ],
        }

        with (
            patch.dict(
                sys.modules,
                {"google": google_module, "google.genai": genai_module},
            ),
            patch.object(directions, "_http_post_json", return_value=fixture),
        ):
            await location_tools.handle_location_tool_call(
                tool_call, session, Location(), "maps-key"
            )

        response = session.responses[0].response
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["steps"][0]["instruction"], "Head north")
        self.assertEqual(response["total_distance"], "500 m")

    async def test_directions_tool_reports_unavailable_gps_without_maps(self) -> None:
        class FakeFunctionResponse:
            def __init__(self, *, id, name, response) -> None:
                self.id = id
                self.name = name
                self.response = response

        class Session:
            def __init__(self) -> None:
                self.responses = []

            async def send_tool_response(self, function_responses) -> None:
                self.responses.extend(function_responses)

        class Location:
            async def current_fix(self):
                raise RuntimeError("GPS is not available on this device yet.")

        google_module = types.ModuleType("google")
        google_module.__path__ = []
        genai_module = types.ModuleType("google.genai")
        genai_module.types = SimpleNamespace(FunctionResponse=FakeFunctionResponse)
        session = Session()
        tool_call = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    id="dir-2",
                    name="get_directions",
                    args={"destination": "the pharmacy"},
                )
            ]
        )

        def fail_if_called(_url: str) -> dict:
            raise AssertionError("Maps must not be called without a GPS fix.")

        with (
            patch.dict(
                sys.modules,
                {"google": google_module, "google.genai": genai_module},
            ),
            patch.object(directions, "_http_post_json", new=fail_if_called),
        ):
            await location_tools.handle_location_tool_call(
                tool_call, session, Location(), "maps-key"
            )

        self.assertEqual(session.responses[0].response["status"], "error")
        self.assertIn("GPS is not available", session.responses[0].response["message"])


if __name__ == "__main__":
    unittest.main()
