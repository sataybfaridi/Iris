
"""Emergency phone calls for IRIS (Twilio, with a mock mode for testing).
 
Calls your emergency contacts in priority order. If the first one doesn't
answer, it tries the next, then the next. It also texts each contact the
location (best effort). This is the "actually call someone" version of the
emergency feature, per the team's decision to use phone calls.
 
Call call_emergency_contacts(location) when the emergency button is pressed.
 
MOCK MODE (no Twilio, no real calls):
    python emergency_calls.py
  It walks through the contacts and asks you "Did they answer?" so you can
  test the escalation logic with no account and no phone. Set it in code with
  mock=True, or in the environment with IRIS_CALLS_MODE=mock.
 
REAL MODE needs these in your environment (.env):
  TWILIO_ACCOUNT_SID   from the Twilio console
  TWILIO_AUTH_TOKEN    from the Twilio console  (SECRET - never commit it)
  TWILIO_FROM_NUMBER   your Twilio trial number, e.g. +1XXXXXXXXXX
  IRIS_USER_NAME       whose device it is (default: "the IRIS user")
  IRIS_CONTACTS_PATH   where contacts.json lives (default: data/contacts.json
                       next to this file -- shared with the website)
 
contacts.json format (a list, in the order to try them):
  [
    {"name": "Mom",    "phone": "+13125550001", "priority": 1},
    {"name": "Dad",    "phone": "+13125550002", "priority": 2},
    {"name": "Sister", "phone": "+13125550003", "priority": 3}
  ]
In mock mode you can add "mock_answer": true/false to a contact to script
who answers instead of being asked.
"""
 
import json
import logging
import os
import re
import time
from pathlib import Path
from xml.sax.saxutils import escape
 
logger = logging.getLogger(__name__)
 
# Where contacts live. Default: a "data" folder next to this file (the same
# place the website saves them), so the app, the website and the alert all
# share ONE file with no setup. Anchored to this file's folder so it works no
# matter which directory you launch from. IRIS_CONTACTS_PATH overrides it.
_DEFAULT_CONTACTS = Path(__file__).resolve().parent / "data" / "contacts.json"
CONTACTS_PATH = Path(os.getenv("IRIS_CONTACTS_PATH", str(_DEFAULT_CONTACTS)))
 
# The device owner's name. It comes from setup: the website saves it to
# account.json right next to contacts.json. IRIS_USER_NAME (env) overrides it
# for testing. This is how a real device gets the name -- entered once during
# setup, not configured by hand.
ACCOUNT_PATH = CONTACTS_PATH.parent / "account.json"
 
RING_SECONDS = 25            # how long to let each phone ring before giving up
POLL_SECONDS = 3             # how often to check the call's status
CALL_TIMEOUT_SECONDS = 60    # hard stop for one call, so we never hang forever
 
 
def _normalize_phone(raw: str):
    """Return a clean +E.164 number (like +13125551234) or None if invalid."""
    digits = re.sub(r"[^\d+]", "", str(raw or ""))
    if digits.startswith("00"):
        digits = "+" + digits[2:]
    if not digits.startswith("+"):
        if len(digits) == 10:
            digits = "+1" + digits
        elif len(digits) == 11 and digits.startswith("1"):
            digits = "+" + digits
        else:
            digits = "+" + digits
    return digits if re.fullmatch(r"\+\d{10,15}", digits) else None
 
 
def _load_contacts(path: Path = CONTACTS_PATH) -> list[dict]:
    """Read contacts.json, keep the ones with a valid phone, sort by priority."""
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError:
        logger.warning("Contacts file not found at %s", path)
        return []
    except (OSError, ValueError) as error:
        logger.warning("Could not read contacts file: %s", error)
        return []
 
    if isinstance(raw, dict):  # tolerate {"contacts":[...]} or {"contact":{...}}
        raw = raw.get("contacts") or ([raw["contact"]] if raw.get("contact") else [raw])
 
    contacts = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        phone = _normalize_phone(item.get("phone", ""))
        name = str(item.get("name", "")).strip()
        if phone:
            entry = {
                "name": name or "your contact",
                "phone": phone,
                "priority": int(item.get("priority", 1) or 1),
            }
            if "mock_answer" in item:          # only used in mock mode
                entry["mock_answer"] = bool(item["mock_answer"])
            contacts.append(entry)
    contacts.sort(key=lambda c: c["priority"])
    return contacts
 
 
def _user_name() -> str:
    """Whose device this is. From IRIS_USER_NAME, else the name saved at
    setup (account.json), else a neutral fallback."""
    name = os.getenv("IRIS_USER_NAME", "").strip()
    if name:
        return name
    try:
        account = json.loads(ACCOUNT_PATH.read_text())
        saved = str(account.get("name", "")).strip() if isinstance(account, dict) else ""
        if saved:
            return saved
    except (OSError, ValueError):
        pass
    return "the IRIS user"
 
 
def _spoken_message(user_name: str, location: str | None) -> str:
    say = f"Hello. This is an emergency alert from {user_name}'s device. {user_name} needs help."
    if location:
        say += f" Their location is {location}."
    say += f" Please check on them now. Again, {user_name} needs help."
    return say
 
 
def _twiml(user_name: str, location: str | None) -> str:
    """The spoken message Twilio reads out when the call connects."""
    say = escape(_spoken_message(user_name, location))  # keep the XML valid
    return f"<Response><Say>{say}</Say><Pause length=\"1\"/><Say>{say}</Say></Response>"
 
 
# ============================================================
# MOCK transport — no Twilio, for testing the logic
# ============================================================
 
def _run_mock(contacts, user_name, location, sms_body, interactive=True):
    print("\n*** MOCK MODE — no real calls or texts are sent ***\n")
    for c in contacts:
        print(f"  TEXT to {c['name']} ({c['phone']}):  {sms_body}")
        print(f"  CALL {c['name']} ({c['phone']}) ... ringing")
        print(f"       voice says: \"{_spoken_message(user_name, location)}\"")
        if "mock_answer" in c:                       # scripted outcome
            answered = c["mock_answer"]
            print(f"       (scripted: {'answers' if answered else 'no answer'})")
        elif interactive:                            # ask at the terminal
            try:
                answered = input(f"       Did {c['name']} answer? [y/N]: ").strip().lower() in ("y", "yes")
            except EOFError:
                answered = False
        else:                                        # running inside the app: don't block on input()
            answered = False
            print("       (no answer — simulated)")
        if answered:
            print(f"  >> {c['name']} ANSWERED. Stopping here.\n")
            return c["name"]
        print(f"  >> {c['name']} did not answer — trying the next contact.\n")
    return None
 
 
# ============================================================
# REAL transport — Twilio
# ============================================================
 
def _wait_for_call(client, call_sid: str) -> str:
    """Poll Twilio until the call reaches a final state (or we time out)."""
    terminal = {"completed", "busy", "no-answer", "failed", "canceled"}
    deadline = time.monotonic() + CALL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            status = client.calls(call_sid).fetch().status
        except Exception as error:
            logger.warning("Could not check call status: %s", error)
            return "unknown"
        if status in terminal:
            return status
        time.sleep(POLL_SECONDS)
    return "timeout"
 
 
def _run_real(contacts, user_name, location, sms_body, sid, token, from_num):
    from twilio.rest import Client
    client = Client(sid, token)
    twiml = _twiml(user_name, location)
    for c in contacts:
        name, phone = c["name"], c["phone"]
        try:  # best-effort text (can be restricted on a trial account)
            client.messages.create(to=phone, from_=from_num, body=sms_body)
            print(f"[Emergency] Texted {name}.", flush=True)
        except Exception as error:
            logger.warning("SMS to %s failed (ok on trial): %s", phone, error)
        try:
            call = client.calls.create(to=phone, from_=from_num, twiml=twiml, timeout=RING_SECONDS)
        except Exception as error:
            logger.warning("Call to %s could not start: %s", phone, error)
            continue
        status = _wait_for_call(client, call.sid)
        print(f"[Emergency] Call to {name}: {status}", flush=True)
        if status == "completed":
            return name
    return None
 
 
# ============================================================
# Public entry point
# ============================================================
 
def call_emergency_contacts(location: str | None = None, reason: str = "",
                            mock=None, interactive: bool = True) -> dict:
    """Call the saved contacts in order until someone answers.
 
    mock=True simulates it (no Twilio). Defaults to env IRIS_CALLS_MODE==mock.
    interactive=False runs the mock without asking "did they answer?" at the
    terminal -- use it when calling from inside the running app.
    From async code, use: await asyncio.to_thread(call_emergency_contacts, location)
    """
    if mock is None:
        mock = os.getenv("IRIS_CALLS_MODE", "").strip().lower() == "mock"
 
    user_name = _user_name()
    contacts = _load_contacts()
    if not contacts:
        return {"status": "no_contacts",
                "message": "No emergency contacts with a phone number are saved."}
 
    sms_body = f"EMERGENCY: {user_name} needs help."
    if reason:
        sms_body += f" ({reason})"
    if location:
        sms_body += f" Location: {location}"
 
    if mock:
        reached = _run_mock(contacts, user_name, location, sms_body, interactive)
    else:
        sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        from_num = os.getenv("TWILIO_FROM_NUMBER", "").strip()
        if not (sid and token and from_num):
            return {"status": "not_configured",
                    "message": "Twilio isn't set up. Check TWILIO_* in your .env, "
                               "or run in mock mode to test the logic."}
        reached = _run_real(contacts, user_name, location, sms_body, sid, token, from_num)
 
    if reached:
        return {"status": "reached",
                "message": f"I reached {reached} and texted everyone the location."}
    return {"status": "no_answer",
            "message": "I called everyone and texted them the location, "
                       "but no one answered the call."}
 
 
if __name__ == "__main__":
    # Runs in MOCK mode so you can test with no Twilio and no phone.
    logging.basicConfig(level=logging.WARNING)
    result = call_emergency_contacts(
        location="41.80, -87.60 (test)", reason="this is a test", mock=True
    )
    print("\nRESULT:", result)