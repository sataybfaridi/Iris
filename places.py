"""Find the nearest place of a given kind, then the walking route to it."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from directions import walking_directions
from interfaces import LocationFix


PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
_FIELD_MASK = "places.displayName,places.formattedAddress"
_SEARCH_RADIUS_METERS = 50_000.0
_QUERY_PREFIXES = ("the nearest ", "nearest ", "a nearby ", "nearby ")

FetchJson = Callable[[str], Awaitable[dict[str, Any]]]


def clean_place_query(query: str) -> str:
    """Drop words like 'nearest' so the search is the kind of place."""
    text = " ".join(query.split())
    lowered = text.lower()
    for prefix in _QUERY_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text


def nearby_request_body(origin: LocationFix, query: str) -> dict[str, Any]:
    """Nearest-place search around a fix. The API key is not in the body."""
    return {
        "textQuery": query,
        "rankPreference": "DISTANCE",
        "maxResultCount": 1,
        "languageCode": "en",
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": origin.latitude,
                    "longitude": origin.longitude,
                },
                "radius": _SEARCH_RADIUS_METERS,
            }
        },
    }


def parse_nearby_place(payload: Any, query: str) -> dict[str, Any]:
    """Turn a Places Text Search body into the closest place, or an error."""
    if not isinstance(payload, dict):
        return {
            "status": "error",
            "message": "Google Maps returned a response that could not be read.",
        }
    if isinstance(payload.get("error"), dict):
        return {
            "status": "error",
            "message": "Google Maps could not search for nearby places.",
        }

    places = payload.get("places") or []
    place = places[0] if places and isinstance(places[0], dict) else None
    if place is None:
        return {
            "status": "error",
            "message": f"No {query} was found near you.",
        }

    display_name = place.get("displayName")
    name = ""
    if isinstance(display_name, dict):
        name = str(display_name.get("text") or "").strip()
    address = str(place.get("formattedAddress") or "").strip()
    if not name and not address:
        return {
            "status": "error",
            "message": f"No {query} was found near you.",
        }
    return {
        "status": "ok",
        "name": name or query,
        "address": address or name,
    }


def _read_json(response: Any) -> dict[str, Any]:
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Places response was not a JSON object.")
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
        return _read_json(error)


async def nearby_place(
    origin: LocationFix,
    query: str,
    api_key: str,
    *,
    fetch: FetchJson | None = None,
) -> dict[str, Any]:
    """Find the closest match and the walking route to its address."""
    kind = clean_place_query(query)
    if not kind:
        return {"status": "error", "message": "Say what kind of place you want."}
    if not api_key.strip():
        return {"status": "error", "message": "Google Maps is not configured."}

    body = nearby_request_body(origin, kind)
    try:
        if fetch is None:
            payload = await asyncio.to_thread(_http_post_json, PLACES_URL, body, api_key)
        else:
            payload = await fetch(PLACES_URL)
    except Exception:
        return {
            "status": "error",
            "message": "Google Maps could not be reached.",
        }

    found = parse_nearby_place(payload, kind)
    if found["status"] != "ok":
        return found

    route = await walking_directions(
        origin, found["address"], api_key, fetch=fetch
    )
    if route.get("status") != "ok":
        return {
            "status": "error",
            "message": route.get("message")
            or "Google Maps could not find walking directions.",
            "name": found["name"],
            "address": found["address"],
        }
    return {
        "status": "ok",
        "name": found["name"],
        "address": found["address"],
        "mode": "walking",
        "total_distance": route.get("total_distance", ""),
        "total_duration": route.get("total_duration", ""),
        "steps": route.get("steps") or [],
    }
