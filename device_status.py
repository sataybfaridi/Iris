"""Live device status for the family dashboard: battery, GPS location, health.

gemini_service.write_status_loop() calls StatusCollector.collect() about every
1.5 s and saves the result to data/status.json. web_server.py serves that file
at /api/status (signed-in accounts only) and the Overview tab draws it.

Every reader here is best-effort. A part that isn't there (fuel gauge not
wired yet, no GPS on this build, running on a laptop) reads as None, and the
dashboard shows a dash for it. Nothing in this file may crash the device.

Battery: MAX17048 fuel gauge (Adafruit 5580) on I2C bus 1 at address 0x36
(SDA = pin 3, SCL = pin 5). Turn I2C on with `sudo raspi-config` ->
Interface Options -> I2C; then `i2cdetect -y 1` should show 36. Reading it
needs the smbus2 package. Without the gauge we fall back to a laptop battery
(psutil), so the page still works when testing on a Mac.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---- Thresholds the dashboard colors by --------------------------------------
LOW_BATTERY_PERCENT = 20
CRITICAL_BATTERY_PERCENT = 10
WARM_CPU_C = 70.0        # the Pi 5 starts slowing itself down around 80-85 C
HOT_CPU_C = 80.0

# ---- MAX17048 fuel gauge -------------------------------------------------------
MAX17048_ADDRESS = 0x36
I2C_BUS = 1
_REG_VCELL = 0x02        # cell voltage, 78.125 microvolts per bit
_REG_SOC = 0x04          # state of charge, 1/256 percent per bit
_REG_VERSION = 0x08      # reads 0x001_ on a MAX17048/49
_REG_CRATE = 0x16        # charge rate, signed, 0.208 percent per hour per bit
_GAUGE_RETRY_SECONDS = 30.0
_RATE_DEADBAND = 1.0     # %/hour; slower than this counts as "not moving"
_MAX_ESTIMATE_MINUTES = 48 * 60

# ---- Linux / Raspberry Pi files ------------------------------------------------
CPU_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
UPTIME_PATH = Path("/proc/uptime")
MEMINFO_PATH = Path("/proc/meminfo")
_POWER_CHECK_SECONDS = 15.0


# ============================================================
# Battery
# ============================================================

def gauge_reading(vcell_raw: int, soc_raw: int, crate_raw: int) -> dict:
    """Turn raw MAX17048 register values into volts, percent and %/hour."""
    if crate_raw & 0x8000:                     # 16-bit two's complement
        crate_raw -= 0x10000
    return {
        "percent": max(0.0, min(100.0, soc_raw / 256.0)),
        "voltage": round(vcell_raw * 78.125e-6, 3),
        "rate_pct_per_hour": round(crate_raw * 0.208, 1),
    }


def battery_level(percent: float) -> str:
    """'ok', 'low' or 'critical' -- the dashboard picks its color from this."""
    if percent <= CRITICAL_BATTERY_PERCENT:
        return "critical"
    if percent <= LOW_BATTERY_PERCENT:
        return "low"
    return "ok"


def _estimate(minutes: float | None) -> int | None:
    if minutes is None or minutes <= 0 or minutes > _MAX_ESTIMATE_MINUTES:
        return None
    return int(round(minutes))


def describe_battery(
    percent: float,
    *,
    source: str,
    rate: float | None = None,
    voltage: float | None = None,
    plugged: bool | None = None,
    seconds_left: float | None = None,
) -> dict:
    """The battery block the dashboard reads."""
    percent = max(0.0, min(100.0, float(percent)))
    shown = int(round(percent))
    minutes_left = minutes_to_full = None
    if rate is not None:                       # fuel gauge: rate says which way
        if rate >= _RATE_DEADBAND:
            state = "charging"
            minutes_to_full = (100.0 - percent) / rate * 60.0
        elif rate <= -_RATE_DEADBAND:
            state = "discharging"
            minutes_left = percent / -rate * 60.0
        else:
            state = "full" if shown >= 99 else "steady"
    elif plugged is not None:                  # laptop battery
        if plugged:
            state = "full" if shown >= 99 else "charging"
        else:
            state = "discharging"
            if seconds_left is not None and seconds_left > 0:
                minutes_left = seconds_left / 60.0
    else:
        state = "unknown"
    return {
        "percent": shown,
        "level": battery_level(percent),
        "state": state,
        "voltage": voltage,
        "rate_pct_per_hour": rate,
        "minutes_left": _estimate(minutes_left),
        "minutes_to_full": _estimate(minutes_to_full),
        "source": source,
    }


def _open_i2c_bus():
    from smbus2 import SMBus                   # Pi only: pip install smbus2
    return SMBus(I2C_BUS)


def _read_register(bus, register: int) -> int:
    high, low = bus.read_i2c_block_data(MAX17048_ADDRESS, register, 2)
    return (high << 8) | low


class BatteryReader:
    """Battery from the MAX17048 fuel gauge, else a laptop battery, else None."""

    def __init__(
        self,
        *,
        bus_factory: Callable[[], Any] | None = None,
        psutil_module: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus_factory = bus_factory or _open_i2c_bus
        self._psutil = psutil_module
        self._clock = clock
        self._bus = None
        self._retry_at = 0.0
        self._said_missing = False

    def read(self) -> dict | None:
        gauge = self._read_gauge()
        if gauge is not None:
            return describe_battery(
                gauge["percent"],
                source="fuel_gauge",
                rate=gauge["rate_pct_per_hour"],
                voltage=gauge["voltage"],
            )
        return self._read_system_battery()

    def close(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass

    def _read_gauge(self) -> dict | None:
        if self._bus is None:
            if self._clock() < self._retry_at:
                return None
            bus = None
            try:
                bus = self._bus_factory()
                version = _read_register(bus, _REG_VERSION)
                if version & 0xFFF0 != 0x0010:
                    raise OSError(f"chip at 0x36 is not a MAX17048 (version {version:#06x})")
            except Exception as error:
                if bus is not None:
                    try:
                        bus.close()
                    except Exception:
                        pass
                self._gauge_missing(error)
                return None
            self._bus = bus
            self._said_missing = False
            logger.info("Battery fuel gauge found on I2C bus %s", I2C_BUS)
        try:
            return gauge_reading(
                _read_register(self._bus, _REG_VCELL),
                _read_register(self._bus, _REG_SOC),
                _read_register(self._bus, _REG_CRATE),
            )
        except Exception as error:
            self.close()
            self._gauge_missing(error)
            return None

    def _gauge_missing(self, error: Exception) -> None:
        self._retry_at = self._clock() + _GAUGE_RETRY_SECONDS
        if not self._said_missing:             # say it once, not every 1.5 s
            self._said_missing = True
            logger.info("Battery fuel gauge not found (%s); trying laptop battery.", error)

    def _read_system_battery(self) -> dict | None:
        try:
            psutil = self._psutil or importlib.import_module("psutil")
            battery = psutil.sensors_battery()
        except Exception:
            return None
        if battery is None or battery.percent is None:
            return None
        return describe_battery(
            battery.percent,
            source="system",
            plugged=battery.power_plugged,
            seconds_left=battery.secsleft,
        )


# ============================================================
# Device health
# ============================================================

def temperature_level(celsius: float | None) -> str | None:
    if celsius is None:
        return None
    if celsius >= HOT_CPU_C:
        return "hot"
    if celsius >= WARM_CPU_C:
        return "warm"
    return "normal"


def parse_throttled(text: str) -> dict | None:
    """Read `vcgencmd get_throttled` output, e.g. 'throttled=0x50000'."""
    _, _, value = text.strip().partition("=")
    try:
        bits = int(value, 16)
    except ValueError:
        return None
    return {
        "under_voltage_now": bool(bits & 0x1),
        "under_voltage_since_boot": bool(bits & 0x10000),
        "slowed_now": bool(bits & 0xE),        # capped, throttled or at the heat limit
    }


def parse_meminfo(text: str) -> float | None:
    """Percent of memory in use, from /proc/meminfo."""
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                values[name] = float(parts[0])
            except ValueError:
                pass
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    if not total or available is None:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - available / total)))


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


class HealthReader:
    """CPU temperature, power-supply warnings, load, memory and uptime."""

    def __init__(
        self,
        *,
        temp_path: Path = CPU_TEMP_PATH,
        uptime_path: Path = UPTIME_PATH,
        meminfo_path: Path = MEMINFO_PATH,
        run: Callable[..., Any] = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
        clock: Callable[[], float] = time.monotonic,
        psutil_module: Any = None,
    ) -> None:
        self._temp_path = temp_path
        self._uptime_path = uptime_path
        self._meminfo_path = meminfo_path
        self._run = run
        self._which = which
        self._clock = clock
        self._psutil = psutil_module
        self._power: dict | None = None
        self._power_checked_at: float | None = None

    def read(self) -> dict:
        temp = self._cpu_temp()
        uptime = self._uptime()
        load = self._load()
        memory = self._memory()
        return {
            "cpu_temp_c": None if temp is None else round(temp, 1),
            "temp_level": temperature_level(temp),
            "uptime_s": None if uptime is None else int(uptime),
            "load_pct": None if load is None else int(round(load)),
            "memory_pct": None if memory is None else int(round(memory)),
            "power": self._power_flags(),
        }

    def _cpu_temp(self) -> float | None:
        text = _read_text(self._temp_path)
        try:
            return int(text.strip()) / 1000.0 if text else None
        except ValueError:
            return None

    def _uptime(self) -> float | None:
        text = _read_text(self._uptime_path)
        if text:
            try:
                return float(text.split()[0])
            except (ValueError, IndexError):
                pass
        psutil = self._get_psutil()            # macOS has no /proc
        try:
            return time.time() - psutil.boot_time() if psutil else None
        except Exception:
            return None

    def _load(self) -> float | None:
        try:
            one_minute = os.getloadavg()[0]
        except (AttributeError, OSError):
            return None
        return min(100.0, 100.0 * one_minute / (os.cpu_count() or 1))

    def _memory(self) -> float | None:
        text = _read_text(self._meminfo_path)
        if text:
            percent = parse_meminfo(text)
            if percent is not None:
                return percent
        psutil = self._get_psutil()
        try:
            return float(psutil.virtual_memory().percent) if psutil else None
        except Exception:
            return None

    def _power_flags(self) -> dict | None:
        """Pi only: under-voltage warnings. Asked every 15 s, not every 1.5 s."""
        now = self._clock()
        if self._power_checked_at is not None and now - self._power_checked_at < _POWER_CHECK_SECONDS:
            return self._power
        self._power_checked_at = now
        self._power = None
        tool = self._which("vcgencmd")
        if not tool:
            return None
        try:
            result = self._run(
                [tool, "get_throttled"], capture_output=True, text=True, timeout=2
            )
            self._power = parse_throttled(result.stdout or "")
        except Exception:
            self._power = None
        return self._power

    def _get_psutil(self):
        if self._psutil is None:
            try:
                self._psutil = importlib.import_module("psutil")
            except Exception:
                self._psutil = False
        return self._psutil or None


# ============================================================
# GPS location
# ============================================================

def _fix_to_dict(fix: Any) -> dict | None:
    try:
        return {
            "latitude": round(float(fix.latitude), 6),
            "longitude": round(float(fix.longitude), 6),
            "accuracy_m": round(float(fix.accuracy_meters), 1),
            "fix_time": float(fix.timestamp),
        }
    except (AttributeError, TypeError, ValueError):
        return None


class LocationTracker:
    """Ask the GPS for a fix without waiting long, and remember the last good
    one, so the dashboard can still show the last known spot when the signal
    drops (indoors, in a tunnel, under heavy tree cover)."""

    def __init__(self, source: Any, *, wait_seconds: float = 0.25) -> None:
        has_fix = callable(getattr(source, "current_fix", None))
        self._source = source if has_fix else None
        self._wait_seconds = wait_seconds
        self._last: dict | None = None

    async def snapshot(self) -> dict:
        if self._source is None:               # this build has no GPS
            return {"state": "unavailable"}
        live = None
        try:
            fix = await asyncio.wait_for(self._source.current_fix(), self._wait_seconds)
            live = _fix_to_dict(fix)
        except Exception:                      # "Waiting for a GPS fix", timeout, unplugged
            live = None
        if live is not None:
            self._last = live
            return {"state": "live", **live}
        if self._last is not None:
            return {"state": "last_known", **self._last}
        return {"state": "searching"}


# ============================================================
# Everything together + saving the file
# ============================================================

class StatusCollector:
    """Battery, location and health -- the parts of status.json besides
    camera, recording and setup mode (gemini_service fills those in)."""

    def __init__(
        self,
        location_source: Any = None,
        *,
        battery: BatteryReader | None = None,
        health: HealthReader | None = None,
    ) -> None:
        self._battery = battery or BatteryReader()
        self._health = health or HealthReader()
        self._location = LocationTracker(location_source)

    async def collect(self) -> dict:
        # I2C, /sys and vcgencmd are blocking calls: keep them off the event
        # loop so the live audio never stutters.
        battery, device = await asyncio.to_thread(self._read_hardware)
        location = await self._location.snapshot()
        return {
            "battery": None if battery is None else battery["percent"],  # old field
            "battery_detail": battery,
            "location": location,
            "device": device,
        }

    def _read_hardware(self) -> tuple[dict | None, dict | None]:
        try:
            battery = self._battery.read()
        except Exception:
            logger.exception("Battery read failed")
            battery = None
        try:
            device = self._health.read()
        except Exception:
            logger.exception("Device health read failed")
            device = None
        return battery, device

    def close(self) -> None:
        self._battery.close()


def write_status_file(path: Path, status: dict) -> None:
    """Save status.json in one step, so the dashboard never reads half a file.
    The file holds the GPS location, so only this user can read it (0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".status-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(status, handle)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
