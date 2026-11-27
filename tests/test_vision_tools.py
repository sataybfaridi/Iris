"""Vision tool dispatch against a Pi-style camera contract."""

import unittest
from types import SimpleNamespace

from vision.tools import VisionState, run_camera_tool


class FakePiStyleCamera:
    def __init__(self, *, on: bool = True) -> None:
        self._enabled = on
        self.zoom_calls = []
        self.focus_calls = 0

    @property
    def is_on(self) -> bool:
        return self._enabled

    async def zoom_to(self, level, center_x=0.5, center_y=0.5, count_as_retry=True):
        self.zoom_calls.append((level, center_x, center_y, count_as_retry))
        return {
            "status": "ok",
            "message": f"Zoomed to {level:g}x",
            "zoom_level": level,
            "frame": SimpleNamespace(jpeg=b"\xff\xd8z\xff\xd9"),
        }

    async def retry_autofocus(self):
        self.focus_calls += 1
        return {
            "status": "ok",
            "message": "Refocused",
            "frame": SimpleNamespace(jpeg=b"\xff\xd8f\xff\xd9"),
        }


class VisionToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_zoom_and_focus_tools_work_with_pi_style_camera(self) -> None:
        camera = FakePiStyleCamera()
        state = VisionState()

        zoom = await run_camera_tool(
            "zoom_camera",
            {"level": 2, "center_x": 0.4, "center_y": 0.6},
            camera,
            state,
        )
        focus = await run_camera_tool("refocus_camera", {}, camera, state)

        self.assertEqual(zoom["status"], "ok")
        self.assertEqual(zoom["zoom_level"], 2)
        self.assertEqual(len(camera.zoom_calls), 1)
        self.assertEqual(focus["status"], "ok")
        self.assertEqual(camera.focus_calls, 1)

    async def test_missing_zoom_methods_are_reported(self) -> None:
        class BareCamera:
            is_on = True

        result = await run_camera_tool(
            "zoom_camera", {"level": 2}, BareCamera(), VisionState()
        )
        self.assertEqual(result["status"], "not_supported")


if __name__ == "__main__":
    unittest.main()
