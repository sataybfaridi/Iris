"""Parse NMEA GGA/RMC sentences from a u-blox NEO-M8N into LocationFix values."""

from __future__ import annotations

import time
from typing import Optional

from interfaces import LocationFix


HDOP_METERS_PER_UNIT = 5.0
MIN_ACCURACY_METERS = 3.0
MAX_ACCURACY_METERS = 100.0
RMC_DEFAULT_ACCURACY_METERS = 30.0


def nmea_checksum_ok(sentence: str) -> bool:
    """True when the sentence has no checksum, or a matching XOR checksum."""
    text = sentence.strip()
    if not text.startswith("$"):
        return False
    if "*" not in text:
        return True
    body, checksum = text[1:].split("*", 1)
    if len(checksum) < 2:
        return False
    try:
        expected = int(checksum[:2], 16)
    except ValueError:
        return False
    value = 0
    for char in body:
        value ^= ord(char)
    return value == expected


def _dm_to_degrees(raw: str, hemisphere: str) -> Optional[float]:
    """Convert NMEA ddmm.mmm / dddmm.mmm plus N/S/E/W into signed degrees."""
    if not raw or not hemisphere:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    degrees = int(value // 100)
    minutes = value - (degrees * 100)
    decimal = degrees + (minutes / 60.0)
    hemi = hemisphere.upper()
    if hemi in ("S", "W"):
        decimal = -decimal
    elif hemi not in ("N", "E"):
        return None
    return decimal


def _accuracy_from_hdop(hdop_raw: str) -> float:
    try:
        hdop = float(hdop_raw)
    except (TypeError, ValueError):
        return RMC_DEFAULT_ACCURACY_METERS
    if hdop <= 0:
        return RMC_DEFAULT_ACCURACY_METERS
    meters = hdop * HDOP_METERS_PER_UNIT
    return max(MIN_ACCURACY_METERS, min(MAX_ACCURACY_METERS, meters))


def _fields(sentence: str) -> list[str] | None:
    text = sentence.strip()
    if not text.startswith("$") or not nmea_checksum_ok(text):
        return None
    body = text[1:]
    if "*" in body:
        body = body.split("*", 1)[0]
    return body.split(",")


def parse_gga(sentence: str, *, timestamp: float | None = None) -> LocationFix | None:
    """Parse a GGA sentence. Returns None when there is no valid fix."""
    fields = _fields(sentence)
    if fields is None or len(fields) < 10:
        return None
    talker = fields[0]
    if not talker.endswith("GGA"):
        return None
    try:
        quality = int(fields[6] or "0")
    except ValueError:
        return None
    if quality <= 0:
        return None
    latitude = _dm_to_degrees(fields[2], fields[3])
    longitude = _dm_to_degrees(fields[4], fields[5])
    if latitude is None or longitude is None:
        return None
    return LocationFix(
        latitude=latitude,
        longitude=longitude,
        accuracy_meters=_accuracy_from_hdop(fields[8]),
        timestamp=time.time() if timestamp is None else timestamp,
    )


def parse_rmc(sentence: str, *, timestamp: float | None = None) -> LocationFix | None:
    """Parse an RMC sentence. Returns None when status is not active."""
    fields = _fields(sentence)
    if fields is None or len(fields) < 7:
        return None
    talker = fields[0]
    if not talker.endswith("RMC"):
        return None
    if (fields[2] or "").upper() != "A":
        return None
    latitude = _dm_to_degrees(fields[3], fields[4])
    longitude = _dm_to_degrees(fields[5], fields[6])
    if latitude is None or longitude is None:
        return None
    return LocationFix(
        latitude=latitude,
        longitude=longitude,
        accuracy_meters=RMC_DEFAULT_ACCURACY_METERS,
        timestamp=time.time() if timestamp is None else timestamp,
    )


def parse_nmea_line(sentence: str, *, timestamp: float | None = None) -> LocationFix | None:
    """Parse one NMEA line. Prefer GGA; fall back to RMC."""
    text = sentence.strip()
    if not text:
        return None
    fields = _fields(text)
    if fields is None:
        return None
    talker = fields[0]
    if talker.endswith("GGA"):
        return parse_gga(text, timestamp=timestamp)
    if talker.endswith("RMC"):
        return parse_rmc(text, timestamp=timestamp)
    return None
