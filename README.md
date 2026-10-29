# Iris-Blind-Companion

Python 3.10+ prototype for a handheld AI companion using the Google GenAI
Gemini Live API. `interfaces.py` defines the async hardware/cloud boundary and
the audio/video frame formats. Domain code is grouped under `location/`,
`vision/`, `gemini/`, and `peripherals/`. Laptop development is supported by
webcam, microphone, speaker, and keyboard drivers in `peripherals/mock.py`.

## Setup

1. Create a virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
3. Keep `DEPLOYMENT_MODE=mock` for laptop development or set it to
   `hardware` for Raspberry Pi 5.
4. Set `GOOGLE_MAPS_API_KEY` for **both** mock and hardware. The key needs
   the Routes API and the Places API (New) enabled.
5. For mock mode, also set `MOCK_LATITUDE`, `MOCK_LONGITUDE`, and
   `MOCK_LOCATION_ACCURACY_METERS`.
6. For hardware mode, set `GPS_SERIAL_PORT` (default `/dev/serial0`) and
   `GPS_BAUD_RATE` (default `9600`) for the u-blox NEO-M8N UART GPS.
7. Run `python main.py`. In mock mode, hold **Space** to talk and hold **C**
   to stream camera frames. Release either key to stop that input.
   Ask "where am I?" to hear the coordinates. Ask for directions to
   a place, or to the nearest gas station, to start turn-by-turn guidance.
   In mock mode Iris simulates walking along the route; on hardware it
   advances from live GPS fixes.

The mock mode sends 16 kHz mono PCM microphone audio and JPEG camera images to
Gemini Live, and plays returned 24 kHz PCM audio through the laptop speaker.
Camera frames are sent at no more than one per second. Keyboard input uses
`pynput`; on some operating systems, global keyboard monitoring may require
accessibility permissions.

## Raspberry Pi hardware

Raspberry Pi hardware mode uses the wired microphone/interface configured as
the ALSA-default input,
the MAX98357A as the ALSA-default I2S output, and the CSI-connected Camera
Module 3 Wide. The provisional button wiring is active-low: talk on BCM GPIO17
and camera on BCM GPIO27, with each button connecting its GPIO pin to ground.
Update `TALK_BUTTON_GPIO` and `CAMERA_BUTTON_GPIO` in
`peripherals/hardware.py` to match the device wiring.

### NEO-M8N GPS (UART)

| NEO-M8N | Pi 5 | Notes |
|---------|------|--------|
| VCC | 3V3 (physical pin 1) | 3.3 V only for this breakout |
| GND | GND (physical pin 6) | |
| TX | GPIO15 / RXD (physical pin 10) | Crossed TX/RX |
| RX | GPIO14 / TXD (physical pin 8) | |
| PPS | leave disconnected | Not used |

OS bring-up on Raspberry Pi OS Bookworm:

1. `sudo raspi-config` → Interface Options → Serial Port: **No** login shell,
   **Yes** serial hardware.
2. In `/boot/firmware/config.txt`, ensure UART is enabled (`enable_uart=1`)
   and free the PL011 if needed (`dtoverlay=disable-bt`).
3. Remove `console=serial0,115200` from `/boot/firmware/cmdline.txt` if present.
4. `sudo systemctl disable hciuart` when Bluetooth was holding the UART, then
   reboot.
5. Confirm `ls -l /dev/serial0` (usually → `/dev/ttyAMA0`).
6. Place the active patch antenna with a clear sky view. A cold start can take
   several minutes before the first valid fix.

Buttons on BCM 17/27 do not conflict with UART on GPIO 14/15. Leave PPS
disconnected so MAX98357A I2S can keep using typical pins (often 18/19/21).

On Raspberry Pi OS, install Picamera2 from the OS packages (`sudo apt install
python3-picamera2`) and use a Python environment that can import system
packages. Configure the Pi's ALSA default capture/playback devices and enable
the MAX98357A I2S overlay before starting with `DEPLOYMENT_MODE=hardware`.

Hardware mode does **not** load OpenCV to start peripherals or capture JPEG
frames from the CSI camera (Picamera2 JPEG / Pillow). If you still use OpenCV
elsewhere on the Pi (for example mock-mode webcam code) and see
`libopenblas.so.0: cannot open shared object file`, install the runtime with
`sudo apt install libopenblas0`.
