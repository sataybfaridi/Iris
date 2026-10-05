# Iris-Blind-Companion

Python 3.10+ scaffold for a handheld AI companion using the Google GenAI
Gemini Live API. `interfaces.py` defines the async hardware/cloud boundary and
the audio/video frame formats. Hardware and laptop device implementations,
along with bidirectional Live streaming, are intentionally left as skeletons.

## Setup

1. Create a virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
3. Keep `DEPLOYMENT_MODE=mock` for laptop development or set it to
   `hardware` for Raspberry Pi 5.
4. Run `python main.py`.

The selected peripheral factory and Gemini streaming loop currently raise
`NotImplementedError` until their driver/transport implementations are added.