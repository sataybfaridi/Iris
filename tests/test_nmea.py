"""NMEA GGA/RMC parsing and NeoM8NLocation behavior without a real UART."""

import asyncio
import time
import unittest

from location.nmea import (
    RMC_DEFAULT_ACCURACY_METERS,
    nmea_checksum_ok,
    parse_gga,
    parse_nmea_line,
    parse_rmc,
)
from peripherals.hardware import NeoM8NLocation


def _checksum(body: str) -> str:
    value = 0
    for char in body:
        value ^= ord(char)
    return f"{value:02X}"


def _sentence(body: str) -> str:
    return f"${body}*{_checksum(body)}"


# Classic valid sentences (checksums verified).
VALID_GGA = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
VALID_RMC = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
VOID_RMC = _sentence("GPRMC,123519,V,,,,,,,230394,,,N")
NO_FIX_GGA = _sentence("GPGGA,123519,,,,,0,00,99.99,,,,,,")


class NmeaParseTests(unittest.TestCase):
    def test_checksum_accepts_valid_sentence(self) -> None:
        self.assertTrue(nmea_checksum_ok(VALID_GGA))
        self.assertFalse(nmea_checksum_ok(VALID_GGA[:-1] + "0"))

    def test_parse_gga_returns_fix_with_hdop_accuracy(self) -> None:
        fix = parse_gga(VALID_GGA, timestamp=10.0)
        self.assertIsNotNone(fix)
        assert fix is not None
        self.assertAlmostEqual(fix.latitude, 48.1173, places=4)
        self.assertAlmostEqual(fix.longitude, 11.5166667, places=4)
        self.assertAlmostEqual(fix.accuracy_meters, 4.5, places=1)
        self.assertEqual(fix.timestamp, 10.0)

    def test_parse_gga_ignores_no_fix(self) -> None:
        self.assertIsNone(parse_gga(NO_FIX_GGA, timestamp=1.0))

    def test_parse_rmc_returns_fix_with_default_accuracy(self) -> None:
        fix = parse_rmc(VALID_RMC, timestamp=11.0)
        self.assertIsNotNone(fix)
        assert fix is not None
        self.assertAlmostEqual(fix.latitude, 48.1173, places=4)
        self.assertEqual(fix.accuracy_meters, RMC_DEFAULT_ACCURACY_METERS)

    def test_parse_rmc_ignores_void_status(self) -> None:
        self.assertIsNone(parse_rmc(VOID_RMC, timestamp=1.0))

    def test_parse_nmea_line_prefers_gga(self) -> None:
        fix = parse_nmea_line(VALID_GGA, timestamp=1.0)
        self.assertIsNotNone(fix)
        assert fix is not None
        self.assertAlmostEqual(fix.accuracy_meters, 4.5, places=1)

    def test_malformed_line_returns_none(self) -> None:
        self.assertIsNone(parse_nmea_line("not nmea"))
        self.assertIsNone(parse_nmea_line(""))


class NeoM8NLocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_injected_gga_becomes_current_fix(self) -> None:
        location = NeoM8NLocation("/dev/null", 9600)
        location.apply_nmea_for_tests(VALID_GGA, timestamp=time.time())
        fix = await location.current_fix()
        self.assertAlmostEqual(fix.latitude, 48.1173, places=4)
        await location.close()

    async def test_void_sentences_do_not_create_a_fix(self) -> None:
        location = NeoM8NLocation("/dev/null", 9600)
        location.apply_nmea_for_tests(VOID_RMC, timestamp=time.time())
        with self.assertRaisesRegex(RuntimeError, "Waiting for a GPS fix"):
            await location.current_fix()
        await location.close()

    async def test_stale_fix_raises_waiting(self) -> None:
        location = NeoM8NLocation(
            "/dev/null", 9600, stale_after_seconds=1.0
        )
        location.apply_nmea_for_tests(VALID_GGA, timestamp=time.time() - 5)
        with self.assertRaisesRegex(RuntimeError, "Waiting for a GPS fix"):
            await location.current_fix()
        await location.close()

    async def test_serial_open_failure_is_speakable(self) -> None:
        def boom(_port: str, _baud: int):
            raise OSError("no such port")

        location = NeoM8NLocation("/dev/does-not-exist", 9600, serial_factory=boom)
        location.start()
        await asyncio.sleep(0.05)
        with self.assertRaisesRegex(RuntimeError, "GPS is disconnected"):
            await location.current_fix()
        await location.close()


if __name__ == "__main__":
    unittest.main()
