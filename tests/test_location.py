"""Mock location, directions parsing, and location tool results."""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_stub)

import config
from directions import (
    ROUTES_URL,
    parse_directions,
    plain_instruction,
    routes_request_body,
    walking_directions,
)
from places import (
    PLACES_URL,
    clean_place_query,
    nearby_place,
    nearby_request_body,
    parse_nearby_place,
)
from interfaces import LocationFix
from location_tools import run_location_tool
from peripherals.mock import LaptopLocation, create_peripherals


DIRECTIONS_FIXTURE = {
    "routes": [
        {
            "distanceMeters": 500,
            "duration": "360s",
            "legs": [
                {
                    "distanceMeters": 500,
                    "duration": "360s",
                    "steps": [
                        {
                            "distanceMeters": 200,
                            "staticDuration": "120s",
                            "navigationInstruction": {
                                "instructions": "Head north on King & Bay"
                            },
                        },
                        {
                            "distanceMeters": 300,
                            "staticDuration": "240s",
                            "navigationInstruction": {
                                "instructions": "Turn left toward the pharmacy"
                            },
                        },
                    ],
                }
            ],
        }
    ],
}


class FakeLocation:
    def __init__(self, fix: LocationFix | None = None, error: Exception | None = None):
        self.fix = fix
        self.error = error

    async def current_fix(self) -> LocationFix:
        if self.error is not None:
            raise self.error
        assert self.fix is not None
        return self.fix


class LocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_location_returns_configured_coordinates(self) -> None:
        location = LaptopLocation(
            latitude=37.7, longitude=-122.4, accuracy_meters=8
        )
        fix = await location.current_fix()
        self.assertEqual(fix.latitude, 37.7)
        self.assertEqual(fix.longitude, -122.4)
        self.assertEqual(fix.accuracy_meters, 8)
        self.assertGreaterEqual(fix.timestamp, 0)
        await location.close()

    def test_plain_instruction_strips_html(self) -> None:
        self.assertEqual(
            plain_instruction("Head <b>north</b> on <b>King &amp; Bay</b>"),
            "Head north on King & Bay",
        )

    def test_parse_directions_returns_plain_steps(self) -> None:
        result = parse_directions(DIRECTIONS_FIXTURE, "the pharmacy")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "walking")
        self.assertEqual(result["total_distance"], "500 m")
        self.assertEqual(result["total_duration"], "6 mins")
        self.assertEqual(
            result["steps"][0]["instruction"],
            "Head north on King & Bay",
        )
        self.assertEqual(result["steps"][1]["distance"], "300 m")
        self.assertEqual(result["steps"][0]["distance_meters"], 200)

    def test_parse_directions_reports_no_route(self) -> None:
        result = parse_directions(
            {"error": {"code": 404, "status": "NOT_FOUND", "message": "No route"}},
            "the pharmacy",
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("No walking route", result["message"])

    async def test_walking_directions_uses_the_fix_as_origin(self) -> None:
        seen = []

        async def fetch(url: str) -> dict:
            seen.append(url)
            return DIRECTIONS_FIXTURE

        origin = LocationFix(37.7, -122.4, 8, timestamp=1)
        result = await walking_directions(
            origin, "the pharmacy", "maps-key", fetch=fetch
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(seen[0], ROUTES_URL)
        body = routes_request_body(origin, "the pharmacy")
        self.assertEqual(body["travelMode"], "WALK")
        self.assertEqual(body["destination"]["address"], "the pharmacy")
        self.assertEqual(
            body["origin"]["location"]["latLng"],
            {"latitude": 37.7, "longitude": -122.4},
        )

    async def test_directions_skip_maps_when_gps_is_unavailable(self) -> None:
        async def fetch(_url: str) -> dict:
            raise AssertionError("Maps must not be called without a GPS fix.")

        result = await run_location_tool(
            "get_directions",
            {"destination": "the pharmacy"},
            FakeLocation(error=RuntimeError("GPS is not available on this device yet.")),
            "maps-key",
            fetch=fetch,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("GPS is not available", result["message"])

    async def test_current_location_tool_returns_the_fix(self) -> None:
        result = await run_location_tool(
            "get_current_location",
            {},
            FakeLocation(LocationFix(37.7, -122.4, 8, timestamp=1)),
            "maps-key",
        )
        self.assertEqual(
            result,
            {
                "status": "ok",
                "latitude": 37.7,
                "longitude": -122.4,
                "accuracy_meters": 8,
            },
        )

    async def test_mock_factory_reads_coordinates_from_settings(self) -> None:
        settings = config.Settings(
            mode=config.DeploymentMode.MOCK,
            gemini_api_key="test-key",
            gemini_live_model="test-model",
            mock_latitude=37.7,
            mock_longitude=-122.4,
            mock_location_accuracy_meters=8,
            google_maps_api_key="maps-key",
        )
        peripherals = create_peripherals(settings)
        fix = await peripherals.location.current_fix()
        self.assertEqual(
            (fix.latitude, fix.longitude, fix.accuracy_meters),
            (37.7, -122.4, 8),
        )
        await peripherals.close()

    def test_clean_place_query_drops_nearest(self) -> None:
        self.assertEqual(clean_place_query("nearest gas station"), "gas station")
        self.assertEqual(clean_place_query("the nearest pharmacy"), "pharmacy")

    def test_parse_nearby_place_returns_the_closest_match(self) -> None:
        result = parse_nearby_place(
            {
                "places": [
                    {
                        "displayName": {"text": "Shell"},
                        "formattedAddress": "123 Main St, New York, NY",
                    }
                ]
            },
            "gas station",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["name"], "Shell")
        self.assertEqual(result["address"], "123 Main St, New York, NY")

    def test_parse_nearby_place_reports_nothing_found(self) -> None:
        result = parse_nearby_place({"places": []}, "gas station")
        self.assertEqual(result["status"], "error")
        self.assertIn("No gas station", result["message"])

    async def test_nearby_place_routes_to_the_found_address(self) -> None:
        seen = []

        async def fetch(url: str) -> dict:
            seen.append(url)
            if url == PLACES_URL:
                return {
                    "places": [
                        {
                            "displayName": {"text": "Shell"},
                            "formattedAddress": "123 Main St, New York, NY",
                        }
                    ]
                }
            return DIRECTIONS_FIXTURE

        origin = LocationFix(40.7128, -74.006, 10, timestamp=1)
        result = await nearby_place(origin, "nearest gas station", "maps-key", fetch=fetch)

        self.assertEqual(seen, [PLACES_URL, ROUTES_URL])
        body = nearby_request_body(origin, "gas station")
        self.assertEqual(body["textQuery"], "gas station")
        self.assertEqual(body["rankPreference"], "DISTANCE")
        self.assertEqual(
            body["locationBias"]["circle"]["center"],
            {"latitude": 40.7128, "longitude": -74.006},
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["name"], "Shell")
        self.assertEqual(result["address"], "123 Main St, New York, NY")
        self.assertEqual(result["steps"][0]["instruction"], "Head north on King & Bay")

    async def test_nearby_tool_skips_search_when_gps_is_unavailable(self) -> None:
        async def fetch(_url: str) -> dict:
            raise AssertionError("Maps must not be called without a GPS fix.")

        result = await run_location_tool(
            "find_nearby_place",
            {"query": "gas station"},
            FakeLocation(error=RuntimeError("GPS is not available on this device yet.")),
            "maps-key",
            fetch=fetch,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("GPS is not available", result["message"])

    def test_hardware_mode_does_not_require_mock_coordinates(self) -> None:
        environment = {
            "DEPLOYMENT_MODE": "hardware",
            "GEMINI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = config.Settings.from_env()
        self.assertIsNone(settings.mock_latitude)
        self.assertEqual(settings.google_maps_api_key, "")


if __name__ == "__main__":
    unittest.main()
