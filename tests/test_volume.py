"""Tests for software playback volume control."""

from __future__ import annotations

import array
import asyncio
import unittest

from gemini_service import listen_for_volume_button
from interfaces import VOLUME_DEFAULT, VOLUME_MAX, VOLUME_MIN, ButtonEvent
from peripherals.hardware import PiAudioOutput
from peripherals.mock import LaptopAudioOutput
from peripherals.volume import VolumeController


class VolumeControllerTests(unittest.TestCase):
    def test_default_level(self) -> None:
        self.assertEqual(VolumeController().get_volume(), VOLUME_DEFAULT)

    def test_adjust_clamps_to_range(self) -> None:
        volume = VolumeController(level=VOLUME_MAX)
        self.assertEqual(volume.adjust_volume(1), VOLUME_MAX)
        volume = VolumeController(level=VOLUME_MIN)
        self.assertEqual(volume.adjust_volume(-1), VOLUME_MIN)

    def test_scale_pcm_silence_at_zero(self) -> None:
        volume = VolumeController(level=0)
        pcm = array.array("h", [1000, -1000]).tobytes()
        self.assertEqual(volume.scale_pcm(pcm), bytes(4))

    def test_scale_pcm_unchanged_at_max(self) -> None:
        volume = VolumeController(level=VOLUME_MAX)
        pcm = array.array("h", [1000, -1000]).tobytes()
        self.assertEqual(volume.scale_pcm(pcm), pcm)

    def test_scale_pcm_half_at_mid(self) -> None:
        volume = VolumeController(level=5)
        pcm = array.array("h", [1000, -1000]).tobytes()
        scaled = array.array("h")
        scaled.frombytes(volume.scale_pcm(pcm))
        self.assertEqual(list(scaled), [500, -500])


class AudioOutputVolumeTests(unittest.TestCase):
    def test_pi_and_laptop_outputs_expose_volume(self) -> None:
        for output in (PiAudioOutput(), LaptopAudioOutput()):
            self.assertEqual(output.get_volume(), VOLUME_DEFAULT)
            self.assertEqual(output.adjust_volume(1), VOLUME_DEFAULT + 1)
            self.assertEqual(output.adjust_volume(-100), VOLUME_MIN)


class QueueButton:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[ButtonEvent] = asyncio.Queue()

    async def push(self, pressed: bool) -> None:
        await self._queue.put(ButtonEvent(pressed=pressed, timestamp=0.0))

    async def events(self):
        while True:
            yield await self._queue.get()


class VolumeButtonListenerTests(unittest.IsolatedAsyncioTestCase):
    async def test_press_adjusts_volume(self) -> None:
        button = QueueButton()
        levels = []

        class FakeOutput:
            def __init__(self) -> None:
                self.level = VOLUME_DEFAULT

            def adjust_volume(self, steps: int) -> int:
                self.level += steps
                levels.append(self.level)
                return self.level

        output = FakeOutput()
        task = asyncio.create_task(
            listen_for_volume_button(button, output, steps=1, label="up")
        )
        await button.push(True)
        await button.push(False)
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(levels, [VOLUME_DEFAULT + 1])


if __name__ == "__main__":
    unittest.main()
