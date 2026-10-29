"""Turn-by-turn walking cues along a route.

The mock device has no live GPS, so a navigation session walks the route at
a normal walking pace and speaks one cue at a time: the current instruction,
then "In 5 feet, ..." just before the next maneuver, then that maneuver.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


FIVE_FEET_METERS = 5 * 0.3048
WALK_METERS_PER_SECOND = 1.4
TICK_SECONDS = 0.5

SpeakCue = Callable[[str], Awaitable[None]]


def approach_phrase(instruction: str) -> str:
    """The warning spoken 5 feet before a maneuver."""
    if not instruction:
        return "In 5 feet, continue."
    lowered = instruction[0].lower() + instruction[1:]
    return f"In 5 feet, {lowered}"


class TurnGuide:
    """Which sentence to speak as distance is added along the steps."""

    def __init__(self, steps: list[tuple[str, float]]) -> None:
        self._steps = [
            (text.strip(), max(float(meters), 0.0))
            for text, meters in steps
            if text and text.strip()
        ]
        self._index = 0
        self._into = 0.0
        self._warned = False
        self.finished = not self._steps

    @classmethod
    def from_route_steps(cls, steps: list[dict[str, Any]]) -> "TurnGuide":
        parsed = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            parsed.append(
                (
                    str(step.get("instruction") or ""),
                    float(step.get("distance_meters") or 0),
                )
            )
        return cls(parsed)

    def start_cues(self) -> list[str]:
        """The only line to speak when navigation begins."""
        if not self._steps:
            self.finished = True
            return ["You have arrived."]
        return [self._steps[0][0]]

    def advance(self, meters: float) -> list[str]:
        """Move forward. Return the cues that become due on this movement."""
        cues: list[str] = []
        remaining = max(0.0, meters)
        while remaining > 0 and not self.finished:
            if self._index >= len(self._steps):
                self.finished = True
                break
            _text, length = self._steps[self._index]
            if length <= 0:
                cues.extend(self._complete_step())
                continue

            upcoming = self._upcoming_phrase()
            warn_at = length - FIVE_FEET_METERS
            if (
                not self._warned
                and upcoming is not None
                and length > FIVE_FEET_METERS
                and self._into < warn_at
            ):
                until_warn = warn_at - self._into
                if remaining < until_warn:
                    self._into += remaining
                    remaining = 0
                    break
                self._into += until_warn
                remaining -= until_warn
                self._warned = True
                cues.append(approach_phrase(upcoming))
                continue

            until_end = length - self._into
            if remaining < until_end:
                self._into += remaining
                remaining = 0
                break
            remaining -= until_end
            cues.extend(self._complete_step())
        return cues

    def _upcoming_phrase(self) -> str | None:
        if self._index + 1 < len(self._steps):
            return self._steps[self._index + 1][0]
        return "you will arrive"

    def _complete_step(self) -> list[str]:
        self._index += 1
        self._into = 0.0
        self._warned = False
        if self._index >= len(self._steps):
            self.finished = True
            if self._steps and "arriv" in self._steps[-1][0].lower():
                return []
            return ["You have arrived."]
        return [self._steps[self._index][0]]


class NavigationController:
    """Walk a route in the background and speak each cue as it comes due."""

    def __init__(self, speak: SpeakCue, idle: asyncio.Event | None = None) -> None:
        self._speak = speak
        self._idle = idle
        self._guide: TurnGuide | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self, steps: list[dict[str, Any]]) -> str:
        """Begin a route. Returns the first instruction and does not speak it."""
        self.cancel()
        guide = TurnGuide.from_route_steps(steps)
        self._guide = guide
        first = guide.start_cues()
        self._task = asyncio.create_task(self._walk(guide), name="navigation")
        return first[0] if first else ""

    def cancel(self) -> None:
        guide = self._guide
        self._guide = None
        if guide is not None:
            guide.finished = True
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()

    async def _walk(self, guide: TurnGuide) -> None:
        try:
            while self._guide is guide and not guide.finished:
                await asyncio.sleep(TICK_SECONDS)
                if self._guide is not guide:
                    return
                for cue in guide.advance(WALK_METERS_PER_SECOND * TICK_SECONDS):
                    if self._idle is not None:
                        await self._idle.wait()
                    if self._guide is not guide:
                        return
                    await self._speak(cue)
        except asyncio.CancelledError:
            return
        except Exception as error:
            print(f"[Navigation] Stopped: {error}", flush=True)
