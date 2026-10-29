"""Turn-by-turn cues: one instruction, then the next as the walk reaches it."""

import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import navigation
from interfaces import LocationFix
from location_tools import handle_location_tool_call, navigation_payload
from navigation import FIVE_FEET_METERS, NavigationController, TurnGuide


class TurnGuideTests(unittest.TestCase):
    def test_start_speaks_only_the_first_instruction(self) -> None:
        guide = TurnGuide([("Head north on Main", 20), ("Turn left onto King", 8)])
        self.assertEqual(guide.start_cues(), ["Head north on Main"])
        self.assertEqual(guide.advance(1), [])

    def test_warns_five_feet_before_the_turn_then_says_the_turn(self) -> None:
        guide = TurnGuide([("Head north", 10), ("Turn left", 6)])
        spoken = guide.start_cues()
        spoken.extend(guide.advance(10 - FIVE_FEET_METERS - 0.1))
        self.assertEqual(spoken, ["Head north"])
        spoken.extend(guide.advance(0.2))
        self.assertEqual(spoken[-1], "In 5 feet, turn left")
        spoken.extend(guide.advance(FIVE_FEET_METERS))
        self.assertEqual(spoken[-1], "Turn left")
        self.assertLess(
            spoken.index("In 5 feet, turn left"),
            spoken.index("Turn left"),
        )

    def test_one_big_step_still_warns_before_the_maneuver(self) -> None:
        guide = TurnGuide([("Head north", 10), ("Turn left", 3)])
        guide.start_cues()
        self.assertEqual(
            guide.advance(10),
            ["In 5 feet, turn left", "Turn left"],
        )

    def test_arrival_is_its_own_cue_at_the_end(self) -> None:
        guide = TurnGuide([("Head north", 10)])
        guide.start_cues()
        cues = []
        while not guide.finished:
            cues.extend(guide.advance(0.5))
        self.assertEqual(cues[0], "In 5 feet, you will arrive")
        self.assertEqual(cues[-1], "You have arrived.")

    def test_payload_hides_the_rest_of_the_route(self) -> None:
        payload = navigation_payload(
            {
                "status": "ok",
                "name": "Shell",
                "address": "123 Main St",
                "total_distance": "200 m",
                "total_duration": "3 mins",
                "steps": [{"instruction": "Head north", "distance_meters": 200}],
            },
            "Head north",
        )
        self.assertEqual(payload["say_now"], "Head north")
        self.assertNotIn("steps", payload)
        self.assertEqual(payload["name"], "Shell")


class NavigationSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_controller_does_not_repeat_the_opening_instruction(self) -> None:
        spoken = []

        async def speak(phrase: str) -> None:
            spoken.append(phrase)

        controller = NavigationController(speak)
        with (
            patch.object(navigation, "TICK_SECONDS", 0.01),
            patch.object(navigation, "WALK_METERS_PER_SECOND", 1000),
        ):
            first = controller.start(
                [
                    {"instruction": "Head north", "distance_meters": 5},
                    {"instruction": "Turn left", "distance_meters": 1},
                ]
            )
            for _ in range(30):
                if any(phrase == "Turn left" for phrase in spoken):
                    break
                await asyncio.sleep(0.01)
            controller.cancel()

        self.assertEqual(first, "Head north")
        self.assertNotIn("Head north", spoken)
        self.assertIn("In 5 feet, turn left", spoken)
        self.assertIn("Turn left", spoken)

    async def test_tool_response_starts_navigation_without_the_step_list(self) -> None:
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
            async def current_fix(self) -> LocationFix:
                return LocationFix(37.7, -122.4, 8, timestamp=1)

        class Navigator:
            def __init__(self) -> None:
                self.steps = None

            def start(self, steps) -> str:
                self.steps = steps
                return steps[0]["instruction"]

        google_module = types.ModuleType("google")
        google_module.__path__ = []
        genai_module = types.ModuleType("google.genai")
        genai_module.types = SimpleNamespace(FunctionResponse=FakeFunctionResponse)
        navigator = Navigator()
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
                            "distanceMeters": 200,
                            "duration": "180s",
                            "steps": [
                                {
                                    "distanceMeters": 200,
                                    "staticDuration": "180s",
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
            patch("directions._http_post_json", return_value=fixture),
        ):
            await handle_location_tool_call(
                tool_call, session, Location(), "maps-key", navigator
            )

        response = session.responses[0].response
        self.assertEqual(response["say_now"], "Head north")
        self.assertNotIn("steps", response)
        self.assertEqual(navigator.steps[0]["instruction"], "Head north")
        self.assertEqual(navigator.steps[0]["distance_meters"], 200)


if __name__ == "__main__":
    unittest.main()
