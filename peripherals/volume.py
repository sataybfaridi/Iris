"""Shared software playback volume for mock and hardware speakers."""

from __future__ import annotations

import array

from interfaces import (
    AUDIO_SAMPLE_WIDTH_BYTES,
    VOLUME_DEFAULT,
    VOLUME_MAX,
    VOLUME_MIN,
)


class VolumeController:
    """Integer volume level applied as PCM gain before speaker write."""

    def __init__(self, level: int = VOLUME_DEFAULT) -> None:
        self._level = _clamp_volume(level)

    def get_volume(self) -> int:
        return self._level

    def adjust_volume(self, steps: int) -> int:
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise TypeError("steps must be an int.")
        self._level = _clamp_volume(self._level + steps)
        return self._level

    def scale_pcm(self, pcm: bytes) -> bytes:
        """Scale signed 16-bit mono PCM by level/VOLUME_MAX."""
        if self._level >= VOLUME_MAX:
            return pcm
        if not pcm:
            return pcm
        usable = len(pcm) - (len(pcm) % AUDIO_SAMPLE_WIDTH_BYTES)
        if usable <= 0:
            return b""
        if self._level <= VOLUME_MIN:
            return bytes(usable)
        samples = array.array("h")
        samples.frombytes(pcm[:usable])
        gain = self._level / VOLUME_MAX
        for index in range(len(samples)):
            samples[index] = int(samples[index] * gain)
        return samples.tobytes()


def _clamp_volume(level: int) -> int:
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError("volume level must be an int.")
    return max(VOLUME_MIN, min(VOLUME_MAX, level))
