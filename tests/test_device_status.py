import asyncio
import json
import os
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_stub)

import device_status
from device_status import (
    BatteryReader,
    HealthReader,
    LocationTracker,
    StatusCollector,
    describe_battery,
    gauge_reading,
    parse_meminfo,
    parse_throttled,
    temperature_level,
    write_status_file,
)


class FakeBus:
    """A pretend MAX17048: 16-bit registers, high byte first."""

    def __init__(self, registers: dict[int, int]) -> None:
        self.registers = registers
        self.closed = False

    def read_i2c_block_data(self, address, register, length):
        assert address == 0x36 and length == 2
        if register not in self.registers:
            raise OSError(121, "Remote I/O error")
        value = self.registers[register]
        return [value >> 8, value & 0xFF]

    def close(self):
        self.closed = True


# version 0x0012, 3.906 V, 76.25 %, -85 * 0.208 = -17.7 %/hour
GAUGE = {0x08: 0x0012, 0x02: 50000, 0x04: 76 * 256 + 64, 0x16: 0x10000 - 85}
NO_LAPTOP_BATTERY = SimpleNamespace(sensors_battery=lambda: None)


def no_bus():
    raise FileNotFoundError("/dev/i2c-1")


class BatteryTests(unittest.TestCase):
    def test_gauge_registers_become_volts_percent_and_rate(self):
        reading = gauge_reading(50000, 76 * 256 + 64, 0x10000 - 85)
        self.assertAlmostEqual(reading["voltage"], 3.906, places=3)
        self.assertAlmostEqual(reading["percent"], 76.25)
        self.assertAlmostEqual(reading["rate_pct_per_hour"], -17.7)

    def test_gauge_percent_never_goes_past_100(self):
        self.assertEqual(gauge_reading(0, 101 * 256, 0)["percent"], 100.0)

    def test_fuel_gauge_is_used_first(self):
        reader = BatteryReader(bus_factory=lambda: FakeBus(GAUGE), psutil_module=NO_LAPTOP_BATTERY)
        battery = reader.read()
        self.assertEqual(battery["source"], "fuel_gauge")
        self.assertEqual(battery["percent"], 76)
        self.assertEqual(battery["level"], "ok")
        self.assertEqual(battery["state"], "discharging")
        self.assertEqual(battery["minutes_left"], 258)
        self.assertIsNone(battery["minutes_to_full"])
        self.assertAlmostEqual(battery["voltage"], 3.906, places=3)

    def test_charging_estimates_time_to_full(self):
        battery = describe_battery(50, source="fuel_gauge", rate=25.0)
        self.assertEqual(battery["state"], "charging")
        self.assertEqual(battery["minutes_to_full"], 120)
        self.assertIsNone(battery["minutes_left"])

    def test_tiny_rate_is_steady_or_full(self):
        self.assertEqual(describe_battery(60, source="fuel_gauge", rate=0.2)["state"], "steady")
        self.assertEqual(describe_battery(100, source="fuel_gauge", rate=0.0)["state"], "full")

    def test_low_and_critical_levels(self):
        self.assertEqual(describe_battery(21, source="x")["level"], "ok")
        self.assertEqual(describe_battery(20, source="x")["level"], "low")
        self.assertEqual(describe_battery(10, source="x")["level"], "critical")

    def test_silly_estimates_are_dropped(self):
        battery = describe_battery(90, source="fuel_gauge", rate=-1.0)   # "90 hours left"
        self.assertIsNone(battery["minutes_left"])

    def test_wrong_chip_at_0x36_is_ignored(self):
        bus = FakeBus({**GAUGE, 0x08: 0x0000})
        reader = BatteryReader(bus_factory=lambda: bus, psutil_module=NO_LAPTOP_BATTERY)
        self.assertIsNone(reader.read())
        self.assertTrue(bus.closed)

    def test_laptop_battery_when_there_is_no_gauge(self):
        laptop = SimpleNamespace(sensors_battery=lambda: SimpleNamespace(
            percent=55.0, secsleft=3600, power_plugged=False))
        battery = BatteryReader(bus_factory=no_bus, psutil_module=laptop).read()
        self.assertEqual(battery["source"], "system")
        self.assertEqual(battery["state"], "discharging")
        self.assertEqual(battery["minutes_left"], 60)

    def test_plugged_in_laptop_is_charging(self):
        laptop = SimpleNamespace(sensors_battery=lambda: SimpleNamespace(
            percent=80.0, secsleft=-2, power_plugged=True))
        battery = BatteryReader(bus_factory=no_bus, psutil_module=laptop).read()
        self.assertEqual(battery["state"], "charging")
        self.assertIsNone(battery["minutes_left"])

    def test_no_battery_anywhere_is_none(self):
        self.assertIsNone(BatteryReader(bus_factory=no_bus, psutil_module=NO_LAPTOP_BATTERY).read())

    def test_missing_gauge_is_retried_every_30_s_not_every_read(self):
        attempts = []

        def missing():
            attempts.append(1)
            raise OSError("no device")

        now = [100.0]
        reader = BatteryReader(bus_factory=missing, psutil_module=NO_LAPTOP_BATTERY, clock=lambda: now[0])
        reader.read()
        reader.read()
        self.assertEqual(len(attempts), 1)
        now[0] += 31
        reader.read()
        self.assertEqual(len(attempts), 2)

    def test_gauge_unplugged_while_running_falls_back(self):
        bus = FakeBus(dict(GAUGE))
        reader = BatteryReader(bus_factory=lambda: bus, psutil_module=NO_LAPTOP_BATTERY)
        self.assertEqual(reader.read()["source"], "fuel_gauge")
        del bus.registers[0x04]
        self.assertIsNone(reader.read())
        self.assertTrue(bus.closed)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "temp").write_text("52125\n")
        (self.dir / "uptime").write_text("8040.55 1234.00\n")
        (self.dir / "meminfo").write_text(
            "MemTotal:        8000000 kB\nMemFree:  1000000 kB\nMemAvailable:    6000000 kB\n")
        self.runs = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_run(self, output="throttled=0x0\n"):
        def run(args, **kwargs):
            self.runs.append(args)
            return SimpleNamespace(stdout=output)
        return run

    def reader(self, **overrides):
        options = dict(
            temp_path=self.dir / "temp",
            uptime_path=self.dir / "uptime",
            meminfo_path=self.dir / "meminfo",
            run=self.fake_run(),
            which=lambda name: "/usr/bin/vcgencmd",
        )
        options.update(overrides)
        return HealthReader(**options)

    def test_reads_the_pi_files(self):
        health = self.reader().read()
        self.assertEqual(health["cpu_temp_c"], 52.1)
        self.assertEqual(health["temp_level"], "normal")
        self.assertEqual(health["uptime_s"], 8040)
        self.assertEqual(health["memory_pct"], 25)
        self.assertEqual(health["power"], {
            "under_voltage_now": False, "under_voltage_since_boot": False, "slowed_now": False})
        self.assertEqual(self.runs, [["/usr/bin/vcgencmd", "get_throttled"]])

    def test_temperature_levels(self):
        self.assertEqual(temperature_level(69.9), "normal")
        self.assertEqual(temperature_level(70.0), "warm")
        self.assertEqual(temperature_level(80.0), "hot")
        self.assertIsNone(temperature_level(None))

    def test_throttle_flags(self):
        flags = parse_throttled("throttled=0x50005")
        self.assertTrue(flags["under_voltage_now"])
        self.assertTrue(flags["under_voltage_since_boot"])
        self.assertTrue(flags["slowed_now"])
        self.assertEqual(parse_throttled("throttled=0x10000"), {
            "under_voltage_now": False, "under_voltage_since_boot": True, "slowed_now": False})
        self.assertIsNone(parse_throttled("VCHI initialization failed"))

    def test_meminfo_without_available_is_unknown(self):
        self.assertIsNone(parse_meminfo("MemTotal: 100 kB\n"))

    def test_power_is_checked_every_15_s(self):
        now = [0.0]
        reader = self.reader(clock=lambda: now[0])
        reader.read()
        reader.read()
        self.assertEqual(len(self.runs), 1)
        now[0] += 16
        reader.read()
        self.assertEqual(len(self.runs), 2)

    def test_laptop_without_pi_files(self):
        laptop = SimpleNamespace(boot_time=lambda: time.time() - 100,
                                 virtual_memory=lambda: SimpleNamespace(percent=33.0))
        health = self.reader(temp_path=self.dir / "missing", uptime_path=self.dir / "missing",
                             meminfo_path=self.dir / "missing", which=lambda name: None,
                             psutil_module=laptop).read()
        self.assertIsNone(health["cpu_temp_c"])
        self.assertIsNone(health["temp_level"])
        self.assertIsNone(health["power"])
        self.assertAlmostEqual(health["uptime_s"], 100, delta=2)
        self.assertEqual(health["memory_pct"], 33)


FIX = SimpleNamespace(latitude=41.834896, longitude=-87.627006, accuracy_meters=3.24, timestamp=1700000000.0)


class FakeGps:
    """Plays back answers to current_fix(): a fix, an error, or 'hang'."""

    def __init__(self, answers):
        self.answers = list(answers)

    async def current_fix(self):
        answer = self.answers.pop(0)
        if answer == "hang":
            await asyncio.sleep(10)
        if isinstance(answer, Exception):
            raise answer
        return answer


class LocationTests(unittest.TestCase):
    def test_build_without_gps(self):
        self.assertEqual(asyncio.run(LocationTracker(None).snapshot()), {"state": "unavailable"})
        self.assertEqual(asyncio.run(LocationTracker(object()).snapshot()), {"state": "unavailable"})

    def test_searching_then_live_then_last_known(self):
        waiting = RuntimeError("Waiting for a GPS fix.")
        tracker = LocationTracker(FakeGps([waiting, FIX, waiting]))

        async def three():
            return [await tracker.snapshot() for _ in range(3)]

        searching, live, last_known = asyncio.run(three())
        self.assertEqual(searching, {"state": "searching"})
        self.assertEqual(live, {"state": "live", "latitude": 41.834896, "longitude": -87.627006,
                                "accuracy_m": 3.2, "fix_time": 1700000000.0})
        self.assertEqual(last_known["state"], "last_known")
        self.assertEqual(last_known["latitude"], 41.834896)

    def test_slow_gps_does_not_hold_up_the_status(self):
        tracker = LocationTracker(FakeGps(["hang"]), wait_seconds=0.05)
        started = time.monotonic()
        snapshot = asyncio.run(tracker.snapshot())
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(snapshot, {"state": "searching"})


class FakeHealth:
    def read(self):
        return {"cpu_temp_c": 50.0, "temp_level": "normal", "uptime_s": 60,
                "load_pct": 5, "memory_pct": 20, "power": None}


class Broken:
    def read(self):
        raise RuntimeError("boom")

    def close(self):
        pass


class CollectorTests(unittest.TestCase):
    def test_status_has_old_and_new_fields(self):
        collector = StatusCollector(
            FakeGps([FIX]),
            battery=BatteryReader(bus_factory=lambda: FakeBus(GAUGE), psutil_module=NO_LAPTOP_BATTERY),
            health=FakeHealth(),
        )
        status = asyncio.run(collector.collect())
        self.assertEqual(status["battery"], 76)                       # old field still there
        self.assertEqual(status["battery_detail"]["source"], "fuel_gauge")
        self.assertEqual(status["location"]["state"], "live")
        self.assertEqual(status["device"]["cpu_temp_c"], 50.0)
        json.dumps(status)                                             # must be JSON-safe

    def test_a_broken_reader_does_not_break_the_status(self):
        collector = StatusCollector(None, battery=Broken(), health=Broken())
        with self.assertLogs("device_status", level="ERROR"):
            status = asyncio.run(collector.collect())
        self.assertIsNone(status["battery"])
        self.assertIsNone(status["battery_detail"])
        self.assertIsNone(status["device"])
        self.assertEqual(status["location"], {"state": "unavailable"})

    def test_status_file_is_whole_and_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data" / "status.json"
            write_status_file(path, {"a": 1})
            write_status_file(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 2})
            self.assertEqual(list(path.parent.glob(".status-*")), [])
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


class StopLoop(Exception):
    pass


class StatusLoopTests(unittest.TestCase):
    def test_loop_writes_device_status(self):
        import gemini_service

        written = []

        def capture(path, status):
            written.append(status)
            raise StopLoop

        peripherals = SimpleNamespace(camera=SimpleNamespace(is_on=True), location=FakeGps([FIX]))
        with patch.object(gemini_service.device_status, "write_status_file", side_effect=capture), \
                patch.object(gemini_service.recording, "is_recording", return_value=False):
            with self.assertRaises(StopLoop):
                asyncio.run(gemini_service.write_status_loop(peripherals, SimpleNamespace(active=True)))
        status = written[0]
        self.assertTrue(status["camera_on"])
        self.assertFalse(status["recording"])
        self.assertTrue(status["editing_allowed"])
        self.assertEqual(status["location"]["state"], "live")
        self.assertIn("battery", status)
        self.assertIn("device", status)
        self.assertAlmostEqual(status["updated"], time.time(), delta=5)


class DashboardTests(unittest.TestCase):
    def test_page_has_the_live_cards(self):
        import web_server

        self.assertNotIn("MOCK", web_server.PAGE)
        for element_id in ("tBatteryMeter", "tLocation", "locMap", "healthList", "hTemp"):
            self.assertIn(f'id="{element_id}"', web_server.PAGE)

    def test_status_needs_sign_in_and_carries_the_device_clock(self):
        import web_server

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with patch.object(web_server, "STATUS_PATH", tmp / "status.json"), \
                    patch.object(web_server, "ACCOUNTS_PATH", tmp / "accounts.json"), \
                    patch.object(web_server, "ACCOUNT_PATH", tmp / "account.json"), \
                    patch.object(web_server, "CONTACTS_PATH", tmp / "contacts.json"), \
                    patch.dict(web_server.SESSIONS, clear=True):
                write_status_file(tmp / "status.json", {"updated": 123.0, "location": {"state": "searching"}})
                account = web_server._create_account(
                    "Jin", "jin@example.com", "secret123", role="admin", can_view_recordings=True)
                token = web_server._new_session(account["id"])
                server = web_server.ThreadingHTTPServer(("127.0.0.1", 0), web_server.Handler)
                threading.Thread(target=server.serve_forever, daemon=True).start()
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                url = f"http://127.0.0.1:{server.server_address[1]}/api/status"
                try:
                    with self.assertRaises(urllib.error.HTTPError) as denied:
                        opener.open(url)
                    self.assertEqual(denied.exception.code, 401)
                    body = json.loads(opener.open(
                        urllib.request.Request(url, headers={"X-Iris-Token": token})).read())
                finally:
                    server.shutdown()
                    server.server_close()
        self.assertEqual(body["location"], {"state": "searching"})
        self.assertEqual(body["updated"], 123.0)
        self.assertAlmostEqual(body["server_time"], time.time(), delta=5)


if __name__ == "__main__":
    unittest.main()
