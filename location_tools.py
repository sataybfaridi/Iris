"""Location tools Gemini can call: the current fix, nearby places, and walking directions.

The camera stays a separate tool. These read CompanionPeripherals.location.
Directions use directions.py. A nearest-place question uses places.py first,
then the walking route to that address.
"""

from __future__ import annotations

import logging
from typing import Any

from directions import walking_directions
from interfaces import Location, LocationFix
from places import nearby_place


logger = logging.getLogger(__name__)

LOCATION_TOOL = "get_current_location"
DIRECTIONS_TOOL = "get_directions"
NEARBY_TOOL = "find_nearby_place"
LOCATION_TOOL_NAMES = frozenset({LOCATION_TOOL, DIRECTIONS_TOOL, NEARBY_TOOL})

LOCATION_INSTRUCTION = """

Location and directions:
- If the user asks where they are, call get_current_location. Then speak \
the latitude, longitude, and accuracy in meters. Do not invent a position.
- If the user asks how to get to a named place, call get_directions with \
that place name. The result contains say_now. Speak only that one sentence, \
like a GPS starting a trip. Do not read later steps, the total distance, \
or a list. Do not invent a route or a street.
- If the user asks for the nearest kind of place, how far it is, or \
directions to it (for example "nearest gas station"), call \
find_nearby_place with the kind of place, such as "gas station". Speak \
the place name and then only say_now. Do not read the rest of the route. \
Do not invent a place.
- A message that starts with "[Navigation cue:" is the next GPS prompt. \
Speak only the sentence after the colon, then stop. Do not add any other \
step or summarize the route.
- If a location, nearby, or directions result has status "error", say that \
message in your own short words. Do not guess a location, a place, or a route.
- Do not read tool names out loud.
"""


def build_location_tools(types: Any) -> Any:
    """The location tool list for LiveConnectConfig(tools=[...])."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=LOCATION_TOOL,
                description=(
                    "Read the device's current GPS fix: latitude, longitude, "
                    "and accuracy in meters. Call this when the user asks "
                    "where they are. Do not invent coordinates."
                ),
            ),
            types.FunctionDeclaration(
                name=DIRECTIONS_TOOL,
                description=(
                    "Start turn-by-turn walking navigation from the current "
                    "GPS fix to a place the user named. Speak only the "
                    "say_now sentence. Later cues arrive on their own. "
                    "Do not invent a route or list the steps."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "destination": types.Schema(
                            type="STRING",
                            description=(
                                "The place the user asked for, in their words, "
                                "such as 'the pharmacy' or 'Union Station'."
                            ),
                        ),
                    },
                    required=["destination"],
                ),
            ),
            types.FunctionDeclaration(
                name=NEARBY_TOOL,
                description=(
                    "Find the closest place of a given kind, such as a gas "
                    "station, and start turn-by-turn walking navigation to "
                    "it. Speak the place name and only the say_now sentence. "
                    "Do not invent a place or list the steps."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description=(
                                "The kind of place, such as 'gas station' or "
                                "'pharmacy'. Do not include the word nearest."
                            ),
                        ),
                    },
                    required=["query"],
                ),
            ),
        ]
    )


def fix_payload(fix: LocationFix) -> dict[str, Any]:
    """The numbers Iris is allowed to speak for a real fix."""
    return {
        "status": "ok",
        "latitude": fix.latitude,
        "longitude": fix.longitude,
        "accuracy_meters": fix.accuracy_meters,
    }


async def _read_fix(location: Location) -> LocationFix | dict[str, Any]:
    try:
        return await location.current_fix()
    except Exception as error:
        message = str(error).strip() or "GPS is not available."
        return {"status": "error", "message": message}


async def run_location_tool(
    name: str,
    args: dict,
    location: Location,
    maps_api_key: str,
    *,
    fetch: Any = None,
) -> dict[str, Any]:
    """Run one location tool. A missing fix never calls Google Maps."""
    if name == LOCATION_TOOL:
        fix = await _read_fix(location)
        if isinstance(fix, dict):
            return fix
        return fix_payload(fix)

    if name == DIRECTIONS_TOOL:
        fix = await _read_fix(location)
        if isinstance(fix, dict):
            return fix
        destination = str((args or {}).get("destination", ""))
        return await walking_directions(
            fix, destination, maps_api_key, fetch=fetch
        )

    if name == NEARBY_TOOL:
        fix = await _read_fix(location)
        if isinstance(fix, dict):
            return fix
        query = str((args or {}).get("query", ""))
        return await nearby_place(fix, query, maps_api_key, fetch=fetch)

    return {"status": "error", "message": f"No tool named {name}."}


def navigation_payload(result: dict[str, Any], say_now: str) -> dict[str, Any]:
    """What Gemini may see: the first cue only, not the rest of the route."""
    payload = {
        "status": "ok",
        "navigation": "started",
        "say_now": say_now,
        "total_distance": result.get("total_distance", ""),
        "total_duration": result.get("total_duration", ""),
        "message": "Speak only say_now. Do not list later steps.",
    }
    if result.get("name"):
        payload["name"] = result["name"]
    if result.get("address"):
        payload["address"] = result["address"]
    return payload


async def handle_location_tool_call(
    tool_call: Any,
    session: Any,
    location: Location,
    maps_api_key: str,
    navigator: Any = None,
) -> None:
    """Answer location tool calls and send the function responses."""
    from google.genai import types

    responses = []
    for call in tool_call.function_calls or ():
        try:
            result = await run_location_tool(
                call.name,
                dict(call.args or {}),
                location,
                maps_api_key,
            )
        except Exception as error:
            logger.warning("Location tool %s failed: %s", call.name, error)
            result = {
                "status": "error",
                "message": "Location help is unavailable right now.",
            }
        steps = result.get("steps") if isinstance(result, dict) else None
        if (
            navigator is not None
            and isinstance(result, dict)
            and result.get("status") == "ok"
            and steps
        ):
            say_now = navigator.start(steps)
            result = navigation_payload(result, say_now)
        print(f"[Location] {call.name}: {result.get('status')}", flush=True)
        responses.append(
            types.FunctionResponse(id=call.id, name=call.name, response=result)
        )

    if responses:
        await session.send_tool_response(function_responses=responses)
