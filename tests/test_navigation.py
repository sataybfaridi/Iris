"""Turn-by-turn cues: one instruction, then the next as the walk reaches it."""

import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from interfaces import LocationFix
from location import navigation
from location.navigation import (
    APPROACH_MIN_STEP_METERS,
    FIVE_FEET_METERS,
    GPS_BACK_CUE,
    GPS_LOST_CUE,
    OFF_ROUTE_CUE,
    NavigationController,
    RouteGeometry,
    TurnGuide,
    decode_polyline,
    haversine_meters,
)
from location.tools import handle_location_tool_call, navigation_payload


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

    def test_short_steps_skip_approach_warning(self) -> None:
        guide = TurnGuide(
            [("Head north", APPROACH_MIN_STEP_METERS - 1), ("Turn left", 6)]
        )
        guide.start_cues()
        cues = guide.advance(APPROACH_MIN_STEP_METERS)
        self.assertNotIn("In 5 feet, turn left", cues)
        self.assertIn("Turn left", cues)

    def test_previews_next_turn_after_a_turn(self) -> None:
        guide = TurnGuide(
            [
                ("Head north", 10),
                ("Turn left onto Eastgate", 40),
                ("Turn right onto Main", 20),
            ]
        )
        guide.start_cues()
        cues = guide.advance(10)
        self.assertIn("Turn left onto Eastgate", cues)
        self.assertNotIn("Next:", "".join(cues))
        cues = guide.advance(10)
        self.assertEqual(cues, ["Next: turn right onto Main"])
        self.assertEqual(guide.advance(5), [])

    def test_previews_arrival_after_last_turn(self) -> None:
        guide = TurnGuide([("Head north", 10), ("Turn left onto Eastgate", 40)])
        guide.start_cues()
        guide.advance(10)
        self.assertEqual(guide.advance(10), ["Next: you will arrive"])

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
        self.assertEqual(
            payload["say_now"],
            "Confirming destination: Shell at 123 Main St. Head north",
        )
        self.assertEqual(payload["destination"], "Shell at 123 Main St")
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
                    {"instruction": "Head north", "distance_meters": 20},
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
                self.destination = None

            def start(self, steps, *, destination: str = "") -> str:
                self.steps = steps
                self.destination = destination
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
            patch("location.directions._http_post_json", return_value=fixture),
        ):
            await handle_location_tool_call(
                tool_call, session, Location(), "maps-key", navigator
            )

        response = session.responses[0].response
        self.assertEqual(
            response["say_now"],
            "Confirming destination: the pharmacy. Head north",
        )
        self.assertEqual(response["destination"], "the pharmacy")
        self.assertNotIn("steps", response)
        self.assertEqual(navigator.steps[0]["instruction"], "Head north")
        self.assertEqual(navigator.steps[0]["distance_meters"], 200)
        self.assertEqual(navigator.destination, "the pharmacy")

    async def test_navigation_status_tool_reports_active_route(self) -> None:
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
            def status(self) -> dict:
                return {
                    "status": "ok",
                    "active": True,
                    "navigation": "on",
                    "destination": "Masjid Haq",
                    "on_route": True,
                    "current_instruction": "Head south on S Stewart Ave",
                    "next_instruction": "Turn right onto Eastgate Rd",
                    "message": "ok",
                }

        google_module = types.ModuleType("google")
        google_module.__path__ = []
        genai_module = types.ModuleType("google.genai")
        genai_module.types = SimpleNamespace(FunctionResponse=FakeFunctionResponse)
        session = Session()
        tool_call = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    id="nav-1",
                    name="get_navigation_status",
                    args={},
                )
            ]
        )

        with patch.dict(
            sys.modules,
            {"google": google_module, "google.genai": genai_module},
        ):
            await handle_location_tool_call(
                tool_call, session, Location(), "maps-key", Navigator()
            )

        response = session.responses[0].response
        self.assertTrue(response["active"])
        self.assertEqual(response["destination"], "Masjid Haq")
        self.assertTrue(response["on_route"])


class RouteGeometryTests(unittest.TestCase):
    def test_decode_polyline_round_trip_sample(self) -> None:
        # Google's documented sample fragment for (38.5, -120.2) -> (40.7, -120.95).
        points = decode_polyline("_p~iF~ps|U_ulLnnqC")
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0], 38.5, places=5)
        self.assertAlmostEqual(points[0][1], -120.2, places=5)

    def test_project_is_monotonic_along_a_straight_route(self) -> None:
        geometry = RouteGeometry([(37.0, -122.0), (37.001, -122.0), (37.002, -122.0)])
        along_a, distance_a = geometry.project(37.0005, -122.0)
        along_b, distance_b = geometry.project(37.0015, -122.0)
        self.assertLess(distance_a, 2.0)
        self.assertLess(distance_b, 2.0)
        self.assertGreater(along_b, along_a)
        self.assertGreater(
            haversine_meters(37.0, -122.0, 37.002, -122.0),
            100,
        )


class GpsNavigationTests(unittest.IsolatedAsyncioTestCase):
    async def test_gps_progress_warns_then_turns_without_rewinding(self) -> None:
        spoken = []

        async def speak(phrase: str) -> None:
            spoken.append(phrase)

        class MovingLocation:
            def __init__(self) -> None:
                self.fixes = [
                    LocationFix(37.0, -122.0, 5, timestamp=1),
                    LocationFix(37.00004, -122.0, 5, timestamp=2),
                    LocationFix(37.00008, -122.0, 5, timestamp=3),
                    LocationFix(37.00012, -122.0, 5, timestamp=4),
                ]
                self.index = 0

            async def current_fix(self) -> LocationFix:
                fix = self.fixes[min(self.index, len(self.fixes) - 1)]
                self.index += 1
                return fix

            async def close(self) -> None:
                return None

        geometry_points = [(37.0, -122.0), (37.00015, -122.0)]
        steps = [
            {
                "instruction": "Head north",
                "distance_meters": 10,
                "encoded_polyline": "",
            },
            {
                "instruction": "Turn left",
                "distance_meters": 6,
                "encoded_polyline": "",
            },
        ]
        location = MovingLocation()
        controller = NavigationController(
            speak, location=location, use_gps=True
        )

        def fake_from_steps(_steps):
            return RouteGeometry(geometry_points)

        with (
            patch.object(navigation, "TICK_SECONDS", 0.01),
            patch.object(navigation, "GPS_START_GRACE_SECONDS", 0),
            patch.object(
                navigation.RouteGeometry, "from_route_steps", fake_from_steps
            ),
        ):
            first = controller.start(steps)
            for _ in range(80):
                if "Turn left" in spoken or OFF_ROUTE_CUE in spoken:
                    break
                await asyncio.sleep(0.01)
            controller.cancel()

        self.assertEqual(first, "Head north")
        self.assertNotIn("Head north", spoken)
        self.assertLessEqual(spoken.count("In 5 feet, turn left"), 1)
        self.assertIn("Turn left", spoken)

    async def test_gps_teleport_resyncs_without_dumping_cues(self) -> None:
        controller = NavigationController(lambda _phrase: None, use_gps=True)
        geometry = RouteGeometry([(37.0, -122.0), (37.001, -122.0)])
        guide = TurnGuide([("Head north", 100), ("Turn left", 10)])
        guide.start_cues()
        # First fix near the start.
        along_a, cues_a = controller._progress_from_fix(
            guide,
            geometry,
            LocationFix(37.0, -122.0, 5, timestamp=1),
            0.0,
        )
        self.assertEqual(cues_a, [])
        # Huge jump along the route: resync, do not fire every intermediate cue.
        along_b, cues_b = controller._progress_from_fix(
            guide,
            geometry,
            LocationFix(37.0008, -122.0, 5, timestamp=2),
            along_a,
        )
        self.assertEqual(cues_b, [])
        self.assertGreater(along_b, along_a)

    async def test_poor_accuracy_does_not_advance(self) -> None:
        controller = NavigationController(lambda _phrase: None, use_gps=True)
        geometry = RouteGeometry([(37.0, -122.0), (37.001, -122.0)])
        guide = TurnGuide([("Head north", 100)])
        guide.start_cues()
        along, cues = controller._progress_from_fix(
            guide,
            geometry,
            LocationFix(37.0005, -122.0, 40, timestamp=1),
            0.0,
        )
        self.assertEqual(along, 0.0)
        self.assertEqual(cues, [])

    async def test_off_route_cancels_with_one_cue(self) -> None:
        spoken = []

        async def speak(phrase: str) -> None:
            spoken.append(phrase)

        class FarLocation:
            async def current_fix(self) -> LocationFix:
                return LocationFix(37.1, -122.0, 5, timestamp=1)

            async def close(self) -> None:
                return None

        steps = [
            {
                "instruction": "Head north",
                "distance_meters": 50,
                "encoded_polyline": "_p~iF~ps|U_ulLnnqC",
            }
        ]
        controller = NavigationController(
            speak, location=FarLocation(), use_gps=True
        )

        def fake_from_steps(_steps):
            return RouteGeometry([(37.0, -122.0), (37.001, -122.0)])

        with (
            patch.object(navigation, "TICK_SECONDS", 0.01),
            patch.object(navigation, "GPS_START_GRACE_SECONDS", 0),
            patch.object(navigation, "OFF_ROUTE_HOLD_SECONDS", 0),
            patch.object(
                navigation.RouteGeometry, "from_route_steps", fake_from_steps
            ),
        ):
            controller.start(steps)
            for _ in range(40):
                if OFF_ROUTE_CUE in spoken:
                    break
                await asyncio.sleep(0.01)
            controller.cancel()

        self.assertEqual(spoken.count(OFF_ROUTE_CUE), 1)

    async def test_brief_off_route_blip_does_not_cancel(self) -> None:
        spoken = []

        async def speak(phrase: str) -> None:
            spoken.append(phrase)

        class BlipLocation:
            def __init__(self) -> None:
                self.n = 0

            async def current_fix(self) -> LocationFix:
                self.n += 1
                # One far fix, then back on the polyline.
                if self.n == 2:
                    return LocationFix(37.1, -122.0, 5, timestamp=self.n)
                return LocationFix(37.00005, -122.0, 5, timestamp=self.n)

            async def close(self) -> None:
                return None

        steps = [
            {
                "instruction": "Head north",
                "distance_meters": 50,
                "encoded_polyline": "",
            }
        ]
        controller = NavigationController(
            speak, location=BlipLocation(), use_gps=True
        )

        def fake_from_steps(_steps):
            return RouteGeometry([(37.0, -122.0), (37.001, -122.0)])

        with (
            patch.object(navigation, "TICK_SECONDS", 0.01),
            patch.object(navigation, "GPS_START_GRACE_SECONDS", 0),
            patch.object(navigation, "OFF_ROUTE_HOLD_SECONDS", 1.0),
            patch.object(
                navigation.RouteGeometry, "from_route_steps", fake_from_steps
            ),
        ):
            controller.start(steps, destination="the park")
            await asyncio.sleep(0.12)
            status = controller.status()
            controller.cancel()

        self.assertNotIn(OFF_ROUTE_CUE, spoken)
        self.assertTrue(status["active"])
        self.assertEqual(status["destination"], "the park")

    async def test_gps_loss_announces_then_recovery(self) -> None:
        spoken = []

        async def speak(phrase: str) -> None:
            spoken.append(phrase)

        class FlakyLocation:
            def __init__(self) -> None:
                self.n = 0

            async def current_fix(self) -> LocationFix:
                self.n += 1
                if self.n <= 8:
                    raise RuntimeError("Waiting for a GPS fix.")
                return LocationFix(37.00005, -122.0, 5, timestamp=self.n)

            async def close(self) -> None:
                return None

        steps = [
            {
                "instruction": "Head north",
                "distance_meters": 50,
                "encoded_polyline": "",
            }
        ]
        controller = NavigationController(
            speak, location=FlakyLocation(), use_gps=True
        )

        def fake_from_steps(_steps):
            return RouteGeometry([(37.0, -122.0), (37.001, -122.0)])

        with (
            patch.object(navigation, "TICK_SECONDS", 0.01),
            patch.object(navigation, "GPS_START_GRACE_SECONDS", 0),
            patch.object(navigation, "GPS_LOST_BEFORE_SPEAK_SECONDS", 0.05),
            patch.object(
                navigation.RouteGeometry, "from_route_steps", fake_from_steps
            ),
        ):
            controller.start(steps)
            for _ in range(80):
                if GPS_BACK_CUE in spoken:
                    break
                await asyncio.sleep(0.01)
            controller.cancel()

        self.assertIn(GPS_LOST_CUE, spoken)
        self.assertIn(GPS_BACK_CUE, spoken)

    def test_status_reports_inactive_when_idle(self) -> None:
        controller = NavigationController(lambda _phrase: None)
        status = controller.status()
        self.assertFalse(status["active"])
        self.assertEqual(status["navigation"], "off")


if __name__ == "__main__":
    unittest.main()
