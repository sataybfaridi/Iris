"""Turn-by-turn walking cues along a route.

Mock mode walks the route at a normal walking pace. Hardware mode projects
live GPS fixes onto the route polyline and advances only when along-route
progress increases.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any

from interfaces import Location, LocationFix


FIVE_FEET_METERS = 5 * 0.3048
WALK_METERS_PER_SECOND = 1.4
TICK_SECONDS = 0.5
OFF_ROUTE_METERS = 25.0
ARRIVAL_METERS = 15.0
EARTH_RADIUS_METERS = 6_371_000.0

SpeakCue = Callable[[str], Awaitable[None]]
OFF_ROUTE_CUE = "You seem off the route. Navigation stopped."


def approach_phrase(instruction: str) -> str:
    """The warning spoken 5 feet before a maneuver."""
    if not instruction:
        return "In 5 feet, continue."
    lowered = instruction[0].lower() + instruction[1:]
    return f"In 5 feet, {lowered}"


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline into (latitude, longitude) points."""
    if not encoded:
        return []
    coordinates: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lng = 0
    length = len(encoded)
    while index < length:
        result = 0
        shift = 0
        while True:
            if index >= length:
                return coordinates
            value = ord(encoded[index]) - 63
            index += 1
            result |= (value & 0x1F) << shift
            shift += 5
            if value < 0x20:
                break
        delta_lat = ~(result >> 1) if result & 1 else (result >> 1)
        lat += delta_lat

        result = 0
        shift = 0
        while True:
            if index >= length:
                return coordinates
            value = ord(encoded[index]) - 63
            index += 1
            result |= (value & 0x1F) << shift
            shift += 5
            if value < 0x20:
                break
        delta_lng = ~(result >> 1) if result & 1 else (result >> 1)
        lng += delta_lng
        coordinates.append((lat / 1e5, lng / 1e5))
    return coordinates


def haversine_meters(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    phi1 = math.radians(latitude_a)
    phi2 = math.radians(latitude_b)
    d_phi = math.radians(latitude_b - latitude_a)
    d_lambda = math.radians(longitude_b - longitude_a)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(a)))


def _project_onto_segment(
    latitude: float,
    longitude: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, float]:
    """Return (along_segment_meters, segment_length_meters, distance_to_segment)."""
    start_lat, start_lon = start
    end_lat, end_lon = end
    segment_length = haversine_meters(start_lat, start_lon, end_lat, end_lon)
    if segment_length <= 0:
        distance = haversine_meters(latitude, longitude, start_lat, start_lon)
        return 0.0, 0.0, distance

    # Local equirectangular projection around the segment start.
    x = math.radians(longitude - start_lon) * math.cos(math.radians(start_lat))
    y = math.radians(latitude - start_lat)
    end_x = math.radians(end_lon - start_lon) * math.cos(math.radians(start_lat))
    end_y = math.radians(end_lat - start_lat)
    dot = x * end_x + y * end_y
    denom = end_x * end_x + end_y * end_y
    t = 0.0 if denom <= 0 else max(0.0, min(1.0, dot / denom))
    along = t * segment_length
    proj_lat = start_lat + t * (end_lat - start_lat)
    proj_lon = start_lon + t * (end_lon - start_lon)
    distance = haversine_meters(latitude, longitude, proj_lat, proj_lon)
    return along, segment_length, distance


class RouteGeometry:
    """Along-route progress from GPS fixes projected onto step polylines."""

    def __init__(self, points: list[tuple[float, float]]) -> None:
        self._points = points
        self._cumulative: list[float] = [0.0]
        total = 0.0
        for index in range(1, len(points)):
            total += haversine_meters(
                points[index - 1][0],
                points[index - 1][1],
                points[index][0],
                points[index][1],
            )
            self._cumulative.append(total)
        self.total_meters = total

    @classmethod
    def from_route_steps(cls, steps: list[dict[str, Any]]) -> "RouteGeometry | None":
        points: list[tuple[float, float]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            encoded = str(step.get("encoded_polyline") or "")
            decoded = decode_polyline(encoded)
            if not decoded:
                continue
            if points and decoded[0] == points[-1]:
                points.extend(decoded[1:])
            else:
                points.extend(decoded)
        if len(points) < 2:
            return None
        return cls(points)

    @property
    def endpoint(self) -> tuple[float, float]:
        return self._points[-1]

    def project(self, latitude: float, longitude: float) -> tuple[float, float]:
        """Return (along_route_meters, distance_to_route_meters)."""
        best_along = 0.0
        best_distance = float("inf")
        for index in range(1, len(self._points)):
            along_segment, _length, distance = _project_onto_segment(
                latitude,
                longitude,
                self._points[index - 1],
                self._points[index],
            )
            along = self._cumulative[index - 1] + along_segment
            if distance < best_distance:
                best_distance = distance
                best_along = along
        return best_along, best_distance


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

    def __init__(
        self,
        speak: SpeakCue,
        idle: asyncio.Event | None = None,
        *,
        location: Location | None = None,
        use_gps: bool = False,
    ) -> None:
        self._speak = speak
        self._idle = idle
        self._location = location
        self._use_gps = use_gps
        self._guide: TurnGuide | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self, steps: list[dict[str, Any]]) -> str:
        """Begin a route. Returns the first instruction and does not speak it."""
        self.cancel()
        guide = TurnGuide.from_route_steps(steps)
        self._guide = guide
        first = guide.start_cues()
        geometry = RouteGeometry.from_route_steps(steps)
        if self._use_gps and self._location is not None and geometry is not None:
            self._task = asyncio.create_task(
                self._follow_gps(guide, geometry),
                name="navigation",
            )
        else:
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

    async def _emit(self, guide: TurnGuide, cue: str) -> None:
        if self._idle is not None:
            await self._idle.wait()
        if self._guide is not guide:
            return
        await self._speak(cue)

    async def _walk(self, guide: TurnGuide) -> None:
        try:
            while self._guide is guide and not guide.finished:
                await asyncio.sleep(TICK_SECONDS)
                if self._guide is not guide:
                    return
                for cue in guide.advance(WALK_METERS_PER_SECOND * TICK_SECONDS):
                    await self._emit(guide, cue)
                    if self._guide is not guide:
                        return
        except asyncio.CancelledError:
            return
        except Exception as error:
            print(f"[Navigation] Stopped: {error}", flush=True)

    async def _follow_gps(self, guide: TurnGuide, geometry: RouteGeometry) -> None:
        last_along = 0.0
        try:
            while self._guide is guide and not guide.finished:
                await asyncio.sleep(TICK_SECONDS)
                if self._guide is not guide:
                    return
                assert self._location is not None
                try:
                    fix = await self._location.current_fix()
                except Exception:
                    continue
                cues = self._progress_from_fix(guide, geometry, fix, last_along)
                if cues is None:
                    await self._emit(guide, OFF_ROUTE_CUE)
                    self.cancel()
                    return
                last_along, due = cues
                for cue in due:
                    await self._emit(guide, cue)
                    if self._guide is not guide:
                        return
        except asyncio.CancelledError:
            return
        except Exception as error:
            print(f"[Navigation] Stopped: {error}", flush=True)

    def _progress_from_fix(
        self,
        guide: TurnGuide,
        geometry: RouteGeometry,
        fix: LocationFix,
        last_along: float,
    ) -> tuple[float, list[str]] | None:
        """Advance from one fix. None means off-route (caller should stop)."""
        along, distance = geometry.project(fix.latitude, fix.longitude)
        end_lat, end_lon = geometry.endpoint
        near_end = (
            haversine_meters(fix.latitude, fix.longitude, end_lat, end_lon)
            <= ARRIVAL_METERS
        )
        if distance > OFF_ROUTE_METERS and not near_end:
            return None

        cues: list[str] = []
        if near_end and not guide.finished:
            remaining = max(0.0, geometry.total_meters - last_along) + 1.0
            cues.extend(guide.advance(remaining))
            if not guide.finished:
                guide.finished = True
                if not cues:
                    cues.append("You have arrived.")
            return geometry.total_meters, cues

        delta = max(0.0, along - last_along)
        if delta > 0:
            cues.extend(guide.advance(delta))
            last_along = along
        return last_along, cues
