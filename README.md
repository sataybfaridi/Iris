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
4. Run `python main.py`. In mock mode, hold **Space** to talk and hold **C**
   to stream camera frames. Release either key to stop that input.

The mock mode sends 16 kHz mono PCM microphone audio and JPEG camera images to
Gemini Live, and plays returned 24 kHz PCM audio through the laptop speaker.
Camera frames are sent at no more than one per second. Keyboard input uses
`pynput`; on some operating systems, global keyboard monitoring may require
accessibility permissions.

Raspberry Pi hardware mode is not implemented yet: the factories in
`peripherals/hardware.py` still raise `NotImplementedError`. Use
`DEPLOYMENT_MODE=mock` until Pi-specific drivers are added.