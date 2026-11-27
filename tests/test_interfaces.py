import unittest

from interfaces import (
    AUDIO_INPUT_SAMPLE_RATE_HZ,
    AUDIO_OUTPUT_SAMPLE_RATE_HZ,
    VIDEO_FRAME_HEIGHT,
    VIDEO_FRAME_WIDTH,
    VIDEO_FRAMES_PER_SECOND,
    AudioInputFrame,
    AudioOutputFrame,
    ButtonEvent,
    LocationFix,
    VideoFrame,
)


class InterfaceContractTests(unittest.TestCase):
    def test_streaming_formats_are_fixed_by_contract(self) -> None:
        self.assertEqual(AUDIO_INPUT_SAMPLE_RATE_HZ, 16_000)
        self.assertEqual(AUDIO_OUTPUT_SAMPLE_RATE_HZ, 24_000)
        self.assertEqual(VIDEO_FRAME_WIDTH, 1280)
        self.assertEqual(VIDEO_FRAME_HEIGHT, 720)
        self.assertEqual(VIDEO_FRAMES_PER_SECOND, 1)

    def test_pcm_frames_accept_valid_16_bit_payloads(self) -> None:
        self.assertEqual(AudioInputFrame(pcm=b"\x00\x00").pcm, b"\x00\x00")
        self.assertEqual(AudioOutputFrame(pcm=b"\x00\x00").pcm, b"\x00\x00")

    def test_pcm_frames_reject_odd_payload_lengths(self) -> None:
        with self.assertRaises(ValueError):
            AudioInputFrame(pcm=b"\x00")
        with self.assertRaises(ValueError):
            AudioOutputFrame(pcm=b"\x00")

    def test_video_frame_requires_complete_jpeg_bytes(self) -> None:
        image = VideoFrame(jpeg=b"\xff\xd8payload\xff\xd9")
        self.assertEqual(image.jpeg, b"\xff\xd8payload\xff\xd9")
        with self.assertRaises(ValueError):
            VideoFrame(jpeg=b"not-a-jpeg")

    def test_button_events_reject_negative_timestamps(self) -> None:
        self.assertEqual(ButtonEvent(pressed=True, timestamp=0.0).pressed, True)
        with self.assertRaises(ValueError):
            ButtonEvent(pressed=False, timestamp=-1.0)

    def test_location_fix_accepts_a_valid_position(self) -> None:
        fix = LocationFix(
            latitude=37.7,
            longitude=-122.4,
            accuracy_meters=5,
            timestamp=10,
        )
        self.assertEqual(fix.latitude, 37.7)
        self.assertEqual(fix.longitude, -122.4)
        self.assertEqual(fix.accuracy_meters, 5)

    def test_location_fix_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            LocationFix(latitude=91, longitude=0, accuracy_meters=1, timestamp=0)
        with self.assertRaises(ValueError):
            LocationFix(latitude=0, longitude=181, accuracy_meters=1, timestamp=0)
        with self.assertRaises(ValueError):
            LocationFix(latitude=0, longitude=0, accuracy_meters=-0.1, timestamp=0)
        with self.assertRaises(ValueError):
            LocationFix(latitude=0, longitude=0, accuracy_meters=1, timestamp=-1)


if __name__ == "__main__":
    unittest.main()
