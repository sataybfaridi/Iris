"""Walking directions from a GPS fix via the Google Maps Routes API."""

from __future__ import annotations

import asyncio
import html
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from interfaces import LocationFix


ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_FIELD_MASK = (
    "routes.duration,routes.distanceMeters,"
    "routes.legs.duration,routes.legs.distanceMeters,"
    "routes.legs.steps.distanceMeters,routes.legs.steps.staticDuration,"
    "routes.legs.steps.navigationInstruction"
)
_TAGS = re.compile(r"<[^>]+>")

FetchJson = Callable[[str], Awaitable[dict[str, Any]]]


def plain_instruction(value: str) -> str:
    """Turn one instruction string into spoken text."""
    text = _TAGS.sub(" ", value)
    text = html.unescape(text).replace("\xa0", " ")
    return " ".join(text.split())


def routes_request_body(origin: LocationFix, destination: str) -> dict[str, Any]:
    """Walking route from a fix to a place name. The API key is not in the body."""
    return {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": origin.latitude,
                    "longitude": origin.longitude,
                }
            }
        },
        "destination": {"address": destination},
        "travelMode": "WALK",
        "languageCode": "en-US",
        "units": "METRIC",
    }


def _spoken_distance(meters: Any) -> str:
    try:
        value = int(meters)
    except (TypeError, ValueError):
        return ""
    if value < 0:
        return ""
    if value < 1000:
        return f"{value} m"
    kilometers = value / 1000
    text = f"{kilometers:.1f}".rstrip("0").rstrip(".")
    return f"{text} km"


def _spoken_duration(value: Any) -> str:
    text = str(value or "")
    if not text.endswith("s"):
        return ""
    try:
        seconds = float(text[:-1])
    except ValueError:
        return ""
    if seconds <= 0:
        return ""
    minutes = int(round(seconds / 60))
    if minutes <= 1:
        return "1 min"
    return f"{minutes} mins"


def parse_directions(payload: Any, destination: str) -> dict[str, Any]:
    """Turn a Routes API JSON body into steps, or a speakable error."""
    if not isinstance(payload, dict):
        return {
            "status": "error",
            "message": "Google Maps returned a response that could not be read.",
        }

    api_error = payload.get("error")
    if isinstance(api_error, dict):
        if api_error.get("status") == "NOT_FOUND":
            return {
                "status": "error",
                "message": f"No walking route was found to {destination}.",
            }
        return {
            "status": "error",
            "message": "Google Maps could not find walking directions.",
        }

    routes = payload.get("routes") or []
    leg = None
    if routes and isinstance(routes[0], dict):
        legs = routes[0].get("legs")
        if legs and isinstance(legs[0], dict):
            leg = legs[0]
    raw_steps = leg.get("steps") if leg else None
    if not routes:
        return {
            "status": "error",
            "message": f"No walking route was found to {destination}.",
        }
    if not leg or not raw_steps:
        return {
            "status": "error",
            "message": "Google Maps did not return any walking steps.",
        }

    steps = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        navigation = step.get("navigationInstruction")
        instruction = ""
        if isinstance(navigation, dict):
            instruction = plain_instruction(str(navigation.get("instructions") or ""))
        if not instruction:
            continue
        meters = step.get("distanceMeters")
        try:
            distance_meters = float(meters) if meters is not None else 0.0
        except (TypeError, ValueError):
            distance_meters = 0.0
        steps.append(
            {
                "instruction": instruction,
                "distance_meters": max(distance_meters, 0.0),
                "distance": _spoken_distance(step.get("distanceMeters")),
                "duration": _spoken_duration(
                    step.get("staticDuration") or step.get("duration")
                ),
            }
        )
    if not steps:
        return {
            "status": "error",
            "message": "Google Maps did not return any walking steps.",
        }

    return {
        "status": "ok",
        "mode": "walking",
        "destination": destination,
        "total_distance": _spoken_distance(leg.get("distanceMeters")),
        "total_duration": _spoken_duration(leg.get("duration")),
        "steps": steps,
    }


def _read_json(response: Any) -> dict[str, Any]:
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Directions response was not a JSON object.")
    return payload


def _http_post_json(url: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": _FIELD_MASK,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            return _read_json(response)
    except HTTPError as error:
        # Permission and routing errors come back as JSON, not an empty failure.
        payload = _read_json(error)
        return payload


async def walking_directions(
    origin: LocationFix,
    destination: str,
    api_key: str,
    *,
    fetch: FetchJson | None = None,
) -> dict[str, Any]:
    """Request a walking route. Errors are a dict the model can speak."""
    place = destination.strip()
    if not place:
        return {"status": "error", "message": "No destination was given."}
    if not api_key.strip():
        return {"status": "error", "message": "Google Maps is not configured."}

    body = routes_request_body(origin, place)
    try:
        if fetch is None:
            payload = await asyncio.to_thread(_http_post_json, ROUTES_URL, body, api_key)
        else:
            payload = await fetch(ROUTES_URL)
    except Exception:
        # The exception text can include the request URL, which holds the key.
        return {
            "status": "error",
            "message": "Google Maps could not be reached.",
        }
    return parse_directions(payload, place)
