"""Camera tools Gemini can call (zoom, refocus) and Gemini's camera rules.

How it works:
1. The user turns the camera on/off with the camera button (a toggle).
2. Gemini looks at the latest picture. If it's clear, it just answers.
3. If the thing is cut off or not in the picture, Gemini tells the user
   which way to move, and the user asks again when ready.
4. If the thing is in the picture but small, Gemini calls zoom_camera.
   If it's blurry, Gemini calls refocus_camera. Each call sends Gemini a
   fresh picture, then a short result.
5. When the retry limits run out, Gemini asks the user to move closer.

The camera is never turned on by a tool. If it's off, Gemini asks the
user to turn it on and point it at the object.

Works with any camera that has new_question(), zoom_to() and
retry_autofocus() (the laptop camera in mock.py and the Pi camera).
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ZOOM_TOOL = "zoom_camera"
FOCUS_TOOL = "refocus_camera"

# Safety net: no matter what Gemini does, at most this many camera tool
# calls per question (the camera also has its own zoom/focus limits).
MAX_TOOL_CALLS_PER_QUESTION = 6

# Save every picture sent to Gemini in debug_frames/ (set IRIS_SAVE_FRAMES=1
# in .env). Off by default: these can be photos of faces or prescriptions.
SAVE_FRAMES_DIR = Path("debug_frames")


_BASE_INSTRUCTION = """\
You are IRIS, a handheld voice assistant for blind and visually impaired \
people. Keep answers short, clear and spoken-friendly. The user cannot see \
the picture, so they fully trust what you say about it.
Reply in English by default. A very short or unclear question that sounds \
like another language is probably misheard English, so still answer in \
English. Only use another language when the user clearly asks for it \
(for example "translate this to Spanish" or "speak Urdu"), or speaks to you \
in full, clear sentences in that language. When asked to translate text in \
a picture, read the original exactly first, then give the translation.

Honesty rules (most important):
- Only describe what is in the most recent camera picture. Never make up \
or guess text, names, numbers or details you can't actually read.
- Never say you "checked again", "looked closely" or "double-checked" \
unless you got a new picture from a camera tool in this answer.
- If the user asks you to check again, look closer, or says you are wrong, \
call zoom_camera or refocus_camera to get a new picture first. Never \
change your answer just because the user disagrees; only change it if the \
new picture shows something different.
- For names, medicine names, doses, numbers and dates: read them exactly. \
If any letter or digit is unclear, say so and spell out only the part you \
can read (for example "it starts with S, A, A, then it's blurry"). \
Getting medicine information wrong can hurt someone.
- If you are not sure, say you are not sure.
- If the picture is dark, black, empty or just a blur, say that instead of \
guessing. Never describe an object you can't clearly see.
- Pictures that arrive while the user is asking show what the camera sees \
right now. Answer from those, not from what you said before.
- Always describe the NEWEST picture. What the user is holding may have \
changed since earlier pictures, so never repeat an old answer about an \
object without looking again.
- If text looks blurry, pixelated or blocky, say you can't read it reliably. \
Never fill in a whole word or name from a few letters; spell only the letters \
you are sure of.
- Never say text that an object "usually" has (like a warning label, brand \
name or expiry date). Only say words you can actually read in this picture. \
If the user insists something is written there and you can't read it, say \
you can't read it. Don't make up text to agree with them.

Camera on and off:
- Most questions don't need the camera (general questions, facts, time, \
chatting). Just answer those. Never ask the user to turn on the camera \
unless they ask about something you need to see.
- The user turns the camera on and off with the camera button. If they ask \
about something you need to see and you have no camera picture, or a tool \
result says the camera is off, ask them to press the camera button and \
point it at the object.
- Every question starts with a "[Camera note: ...]" saying if the camera is \
ON or OFF. Trust it. If it says OFF, you can't see anything right now: never \
describe the user, the room or any object from earlier pictures. Never \
read the camera note out loud.

Guiding the user:
- If the thing the user asks about is cut off, not in the picture, too \
close, or at a bad angle, tell them how to move the camera in ONE short \
step: left, right, up, down, closer, farther, or tilt. For example: \
"Move it a little to the left." Then say "Tell me when you're ready."
- After asking the user to move, do not answer from the old picture. Wait \
until they ask again, then use the newest picture.
- If the thing is in the picture but just small, zoom first. Only ask the \
user to move closer if zooming doesn't help.
{direction_note}

Zoom and focus:
- If the picture is clear enough to answer, just answer. Do not zoom or \
refocus for no reason.
- If the picture is blurry, call refocus_camera once, then look at the new \
picture.
- If the thing is in the picture but too small to read, call zoom_camera. \
The level is the TOTAL zoom compared to the normal view, not extra zoom: \
if you are at 2 and want closer, ask for 3 or 4, never 2 again. Every tool \
result tells you the current zoom_level. Set center_x and center_y to \
where the thing is in the picture you are looking at right now \
(0 = left/top, 1 = right/bottom, 0.5 = middle).
- After every camera tool call you get a new picture. Look at it before \
answering.
- If a tool result has ask_user_to_move_closer true, or the new picture is \
still unclear after the last try, guide the user closer.
- You can say something short like "let me look closer" while zooming. \
Don't read out zoom levels or tool names.
"""

_FACES_AWAY_NOTE = (
    "- The camera points away from the user, like a phone's back camera, so "
    "left in the picture is the user's left."
)
_FACES_USER_NOTE = (
    "- This camera faces the user (a laptop webcam), so the picture is like "
    "looking at them: left in the picture is the USER'S RIGHT. Flip left and "
    "right when you give directions."
)


def build_system_instruction(camera_faces_user: bool) -> str:
    """Gemini's instructions. The laptop webcam faces you (directions flip);
    the Pi camera points away from you like a phone's back camera."""
    from location_tools import LOCATION_INSTRUCTION

    note = _FACES_USER_NOTE if camera_faces_user else _FACES_AWAY_NOTE
    return _BASE_INSTRUCTION.format(direction_note=note) + LOCATION_INSTRUCTION


# Default (Pi-style camera). gemini_service picks the right one.
SYSTEM_INSTRUCTION = build_system_instruction(camera_faces_user=False)


def build_camera_tools(types: Any) -> Any:
    """The tool list for LiveConnectConfig(tools=[...])."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=ZOOM_TOOL,
                description=(
                    "Digitally zoom the camera when the thing is in the "
                    "picture but too small to read, then get a new picture. "
                    "Only works while the camera is on."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "level": types.Schema(
                            type="NUMBER",
                            description=(
                                "TOTAL zoom compared to the normal view: "
                                "1, 2, 3 or 4. Not extra zoom. To get closer "
                                "than the current zoom_level, use a bigger number."
                            ),
                        ),
                        "center_x": types.Schema(
                            type="NUMBER",
                            description=(
                                "Where the thing is left-to-right in the "
                                "current picture, 0.0 to 1.0."
                            ),
                        ),
                        "center_y": types.Schema(
                            type="NUMBER",
                            description=(
                                "Where the thing is top-to-bottom in the "
                                "current picture, 0.0 to 1.0."
                            ),
                        ),
                    },
                    required=["level", "center_x", "center_y"],
                ),
            ),
            types.FunctionDeclaration(
                name=FOCUS_TOOL,
                description=(
                    "Run autofocus once when the picture is blurry, then "
                    "get a new picture. Only works while the camera is on."
                ),
            ),
        ]
    )


@dataclass
class VisionState:
    """What the Gemini side remembers about the camera."""

    # The view Gemini saw last, so its coordinates can be converted
    # (Gemini gives positions in the picture it's looking at, which might
    # already be zoomed in).
    last_zoom: float = 1.0
    last_center_x: float = 0.5
    last_center_y: float = 0.5
    tool_calls_this_question: int = 0
    # Last time the user did something (camera on, Talk, camera tool).
    # The camera turns itself off after a while with no activity.
    last_activity: float = 0.0

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def reset_view(self) -> None:
        """New question: back to the normal view, tool-call count reset."""
        self.tool_calls_this_question = 0
        self.last_zoom = 1.0
        self.last_center_x = 0.5
        self.last_center_y = 0.5

    def to_full_view(self, x: float, y: float) -> tuple[float, float]:
        """Point in the current (maybe zoomed) picture -> point in the 1x view."""
        full_x = self.last_center_x + (x - 0.5) / self.last_zoom
        full_y = self.last_center_y + (y - 0.5) / self.last_zoom
        return full_x, full_y


def save_debug_frame(frame: Any, label: str = "frame") -> None:
    """Save a picture to debug_frames/ if IRIS_SAVE_FRAMES=1 is in .env."""
    if frame is None or os.getenv("IRIS_SAVE_FRAMES", "").strip() != "1":
        return
    jpeg = getattr(frame, "jpeg", frame)
    try:
        SAVE_FRAMES_DIR.mkdir(exist_ok=True)
        name = time.strftime("%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}_{label}.jpg"
        (SAVE_FRAMES_DIR / name).write_bytes(jpeg)
    except OSError as error:
        logger.warning("Could not save debug frame: %s", error)


def _number(args: dict, key: str, default: float) -> float:
    try:
        return float(args.get(key, default))
    except (TypeError, ValueError):
        return default


def _camera_is_on(camera: Any) -> bool:
    return bool(getattr(camera, "is_on", getattr(camera, "_enabled", False)))


async def run_camera_tool(
    name: str, args: dict, camera: Any, state: VisionState
) -> dict:
    """Run one camera tool. Returns the result dict (may contain "frame")."""
    if name not in (ZOOM_TOOL, FOCUS_TOOL):
        return {"status": "unknown_tool", "message": f"No tool named {name}."}

    if not hasattr(camera, "zoom_to") or not hasattr(camera, "retry_autofocus"):
        return {
            "status": "not_supported",
            "message": "This camera can't zoom or refocus. Guide the user closer instead.",
            "ask_user_to_move_closer": True,
        }

    # Never turn the camera on by ourselves: the user might not be pointing
    # it at anything yet, and those pictures would mean nothing.
    if not _camera_is_on(camera):
        return {
            "status": "camera_off",
            "message": "The camera is off. Ask the user to press the camera "
                       "button and point it at the object, then ask again.",
            "ask_user_to_turn_on_camera": True,
        }

    if state.tool_calls_this_question >= MAX_TOOL_CALLS_PER_QUESTION:
        return {
            "status": "limit_reached",
            "message": "No more camera tries for this question. "
                       "Guide the user to move closer or re-aim the camera.",
            "ask_user_to_move_closer": True,
        }
    state.tool_calls_this_question += 1
    state.touch()

    if name == FOCUS_TOOL:
        print("[Vision] Gemini asked to refocus.", flush=True)
        return await camera.retry_autofocus()

    level = _number(args, "level", 2.0)
    x, y = state.to_full_view(
        _number(args, "center_x", 0.5),
        _number(args, "center_y", 0.5),
    )
    print(f"[Vision] Gemini asked to zoom {level:g}x.", flush=True)
    result = await camera.zoom_to(level, x, y)

    # Same zoom on the same spot doesn't use a camera retry, but Gemini
    # still gets a fresh picture so "check again" really checks again.
    if result.get("status") == "no_change" and result.get("frame") is None:
        fresh = getattr(camera, "_fresh_frame", None)
        if fresh is not None:
            result["frame"] = await fresh()

    if result.get("status") == "ok":
        state.last_zoom = float(result.get("zoom_level", level))
        state.last_center_x = min(1.0, max(0.0, x))
        state.last_center_y = min(1.0, max(0.0, y))
    return result


async def handle_tool_call(
    tool_call: Any, session: Any, camera: Any, state: VisionState
) -> None:
    """Answer one Gemini tool call: send new pictures, then the results."""
    from google.genai import types

    responses = []
    for call in tool_call.function_calls or ():
        try:
            result = await run_camera_tool(
                call.name, dict(call.args or {}), camera, state
            )
        except Exception as error:  # never leave Gemini waiting
            logger.warning("Camera tool %s failed: %s", call.name, error)
            result = {
                "status": "error",
                "message": "The camera had a problem. Guide the user closer.",
                "ask_user_to_move_closer": True,
            }

        frame = result.pop("frame", None)
        if frame is not None:
            save_debug_frame(frame, call.name)
            # Send the new picture first so Gemini has it when it reads the result.
            await session.send_realtime_input(
                video=types.Blob(data=frame.jpeg, mime_type="image/jpeg")
            )
        print(f"[Vision] {call.name}: {result.get('status')}", flush=True)
        responses.append(
            types.FunctionResponse(id=call.id, name=call.name, response=result)
        )

    if responses:
        await session.send_tool_response(function_responses=responses)