# Iris-Blind-Companion

Python 3.10+ prototype for a handheld AI companion using the Google GenAI
Gemini Live API. `interfaces.py` defines the async hardware/cloud boundary and
the audio/video frame formats. Laptop development is supported by webcam,
microphone, speaker, and keyboard drivers in `peripherals/mock.py`.

## Setup

1. Create a virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
3. Keep `DEPLOYMENT_MODE=mock` for laptop development or set it to
   `hardware` for Raspberry Pi 5.
4. Run `python main.py`. In mock mode, hold **Space** to talk and press **C**
   to toggle the camera. Press **Up** / **Down** for volume. Release Talk to
   stop sending microphone audio.

The mock mode sends 16 kHz mono PCM microphone audio and JPEG camera images to
Gemini Live, and plays returned 24 kHz PCM audio through the laptop speaker.
Camera frames are sent at no more than one per second. Keyboard input uses
`pynput`; on some operating systems, global keyboard monitoring may require
accessibility permissions.

Raspberry Pi hardware mode uses the wired microphone/interface configured as
the ALSA-default input,
the MAX98357A as the ALSA-default I2S output, and the CSI-connected Camera
Module 3 Wide. The provisional button wiring is active-low, each button
connecting its GPIO pin to ground:

| Button | BCM GPIO | Physical pin |
|--------|----------|--------------|
| Talk | 17 | 11 |
| Camera | 27 | 13 |
| Volume up | 23 | 16 |
| Volume down | 24 | 18 |

Update the `*_BUTTON_GPIO` constants in `peripherals/hardware.py` if your
wiring differs. Volume is software gain on playback (levels 0–10).

On Raspberry Pi OS, install Picamera2 from the OS packages (`sudo apt install
python3-picamera2`) and use a Python environment that can import system
packages. Configure the Pi's ALSA default capture/playback devices and enable
the MAX98357A I2S overlay before starting with `DEPLOYMENT_MODE=hardware`.

Hardware mode does **not** load OpenCV to start peripherals or capture JPEG
frames from the CSI camera (Picamera2 JPEG / Pillow). If you still use OpenCV
elsewhere on the Pi (for example mock-mode webcam code) and see
`libopenblas.so.0: cannot open shared object file`, install the runtime with
`sudo apt install libopenblas0`.
