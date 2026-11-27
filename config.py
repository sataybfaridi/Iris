"""Environment-backed application configuration."""

from dataclasses import dataclass
from enum import Enum
import os

from dotenv import load_dotenv


class DeploymentMode(str, Enum):
    MOCK = "mock"
    HARDWARE = "hardware"


DEFAULT_GPS_SERIAL_PORT = "/dev/serial0"
DEFAULT_GPS_BAUD_RATE = 9600


def _required_float(name: str, low: float, high: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        raise ValueError(
            f"{name} must be set before starting the companion in mock mode."
        )
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}.")
    return value


def _gps_baud_rate() -> int:
    raw = os.getenv("GPS_BAUD_RATE", str(DEFAULT_GPS_BAUD_RATE)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("GPS_BAUD_RATE must be an integer.") from error
    if value <= 0:
        raise ValueError("GPS_BAUD_RATE must be greater than zero.")
    return value


@dataclass(frozen=True)
class Settings:
    mode: DeploymentMode
    gemini_api_key: str
    gemini_live_model: str
    mock_latitude: float | None = None
    mock_longitude: float | None = None
    mock_location_accuracy_meters: float | None = None
    google_maps_api_key: str = ""
    gps_serial_port: str = DEFAULT_GPS_SERIAL_PORT
    gps_baud_rate: int = DEFAULT_GPS_BAUD_RATE

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        mode_value = os.getenv("DEPLOYMENT_MODE", DeploymentMode.MOCK.value).strip().lower()
        try:
            mode = DeploymentMode(mode_value)
        except ValueError as error:
            supported_modes = ", ".join(item.value for item in DeploymentMode)
            raise ValueError(
                f"DEPLOYMENT_MODE must be one of: {supported_modes}"
            ) from error

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set before starting the companion.")

        model = os.getenv(
            "GEMINI_LIVE_MODEL",
            "gemini-3.8-live",
        ).strip()
        if not model:
            raise ValueError("GEMINI_LIVE_MODEL must not be empty.")

        mock_latitude = None
        mock_longitude = None
        mock_accuracy = None
        maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
        gps_port = DEFAULT_GPS_SERIAL_PORT
        gps_baud = DEFAULT_GPS_BAUD_RATE

        if mode is DeploymentMode.MOCK:
            mock_latitude = _required_float("MOCK_LATITUDE", -90, 90)
            mock_longitude = _required_float("MOCK_LONGITUDE", -180, 180)
            mock_accuracy = _required_float("MOCK_LOCATION_ACCURACY_METERS", 0, 1_000_000)
            if not maps_key:
                raise ValueError(
                    "GOOGLE_MAPS_API_KEY must be set before starting the "
                    "companion in mock mode."
                )
        elif mode is DeploymentMode.HARDWARE:
            if not maps_key:
                raise ValueError(
                    "GOOGLE_MAPS_API_KEY must be set before starting the "
                    "companion in hardware mode."
                )
            gps_port = (
                os.getenv("GPS_SERIAL_PORT", DEFAULT_GPS_SERIAL_PORT).strip()
                or DEFAULT_GPS_SERIAL_PORT
            )
            gps_baud = _gps_baud_rate()

        return cls(
            mode=mode,
            gemini_api_key=api_key,
            gemini_live_model=model,
            mock_latitude=mock_latitude,
            mock_longitude=mock_longitude,
            mock_location_accuracy_meters=mock_accuracy,
            google_maps_api_key=maps_key,
            gps_serial_port=gps_port,
            gps_baud_rate=gps_baud,
        )
