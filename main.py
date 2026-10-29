"""Application entry point for the handheld companion prototype."""

import asyncio

from config import DeploymentMode, Settings
from gemini.service import run_gemini_live
from interfaces import CompanionPeripherals


def create_peripherals(settings: Settings) -> CompanionPeripherals:
    if settings.mode is DeploymentMode.HARDWARE:
        from peripherals.hardware import create_peripherals as create_hardware

        return create_hardware(settings)

    from peripherals.mock import create_peripherals as create_mock

    return create_mock(settings)


async def run() -> None:
    settings = Settings.from_env()
    peripherals = create_peripherals(settings)
    try:
        await peripherals.start()
        await run_gemini_live(settings, peripherals)
    finally:
        await peripherals.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
