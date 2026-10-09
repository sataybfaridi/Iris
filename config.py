"""Environment-backed application configuration."""

from dataclasses import dataclass
from enum import Enum
import os

from dotenv import load_dotenv


class DeploymentMode(str, Enum):
    MOCK = "mock"
    HARDWARE = "hardware"


@dataclass(frozen=True)
class Settings:
    mode: DeploymentMode
    gemini_api_key: str
    gemini_live_model: str

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
            "gemini_3.8_live",
        ).strip()
        if not model:
            raise ValueError("GEMINI_LIVE_MODEL must not be empty.")

        return cls(
            mode=mode,
            gemini_api_key=api_key,
            gemini_live_model="gemini_3.8_live",
        )
