"""IRIS setup-mode website -- the local dashboard you open in a browser.
 
This is the web page that opens when the user holds Camera + Talk to enter
setup mode. Run it on your laptop and go to http://localhost:5000 -- you can
sign up, log in, and add emergency contacts. The contacts you add here are
saved to the SAME file emergency_calls.py reads (~/.iris/contacts.json), so
adding a contact in the browser means the emergency feature will call them.
 
Run it:
    python web_server.py
Then open your browser to:
    http://localhost:5000
 
No Flask, no pip install -- only Python's built-in modules. Free, runs off
your own laptop. (It's only reachable on your machine while it's running,
which is exactly how a device's own setup page works.)
"""
 
import json
import os
import re
import hashlib
import secrets
import socket
import logging
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
 
# Reuse the real emergency code so the website and the alert feature agree on
# where contacts live and how phone numbers are cleaned up.
import emergency_calls
# The dashcam recorder -- we read its saved clip list to show on the dashboard.
import recording
 
logger = logging.getLogger(__name__)
 
# Listen on 0.0.0.0 so other devices on the SAME WiFi can reach the dashboard
# at this machine's local IP (not just this machine's own localhost). Family on
# the same network can open it; it is still NOT on the public internet.
HOST = "0.0.0.0"
PORT = int(os.getenv("IRIS_WEB_PORT", "5000"))
 
CONTACTS_PATH = emergency_calls.CONTACTS_PATH            # data/contacts.json
ACCOUNT_PATH = CONTACTS_PATH.parent / "account.json"     # old single account (migrated)
ACCOUNTS_PATH = CONTACTS_PATH.parent / "accounts.json"   # all accounts (admin + viewers)
STATUS_PATH = CONTACTS_PATH.parent / "status.json"       # live device status (app writes it)
 
 
MAX_CONTACTS = 5          # most emergency contacts the device will keep/call
 
 
def _read_status() -> dict:
    """The device's live status, written by the running app."""
    try:
        return json.loads(STATUS_PATH.read_text())
    except (OSError, ValueError):
        return {}
 
 
def _editing_allowed() -> bool:
    """Contacts can be edited only while the device is in setup mode. If there
    is no status file at all (app not running, e.g. standalone testing), don't
    lock, so the page still works on its own."""
    status = _read_status()
    if not status:
        return True
    return bool(status.get("editing_allowed", False))
 
# Logged-in browser sessions (token -> account id). In memory only: restarting
# the server logs everyone out. We store the id (not the role) so permission
# changes take effect immediately, without the person signing in again.
SESSIONS: dict[str, int] = {}
 
 
# ============================================================
# Storage helpers (contacts + the one account)
# ============================================================
 
def _ensure_dir() -> None:
    CONTACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
 
 
def _load_contacts() -> list[dict]:
    """Read contacts.json as a plain list of dicts (keeps id/name/phone/priority)."""
    try:
        raw = json.loads(CONTACTS_PATH.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return []
    if isinstance(raw, dict):                      # tolerate {"contacts":[...]}
        raw = raw.get("contacts") or []
    out = []
    for i, item in enumerate(raw if isinstance(raw, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        out.append({
            "id": item.get("id") or i,
            "name": str(item.get("name", "")).strip(),
            "phone": str(item.get("phone", "")).strip(),
            "priority": int(item.get("priority", 1) or 1),
        })
    out.sort(key=lambda c: c["priority"])
    return out
 
 
def _save_contacts(contacts: list[dict]) -> None:
    """Write contacts.json in the format emergency_calls.py expects, chmod 600."""
    _ensure_dir()
    CONTACTS_PATH.write_text(json.dumps(contacts, indent=2))
    try:
        os.chmod(CONTACTS_PATH, 0o600)             # keep contacts private
    except OSError:
        pass
 
 
def _hash_password(password: str, salt: str) -> str:
    """Salted PBKDF2 hash -- we never store the raw password."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100_000
    ).hex()
 
 
# ---- Accounts: one admin (full control) + viewers (limited) --------------
#
# Each account: {id, name, email, salt, hash, role, can_view_recordings}.
#   role "admin"  -> full control: edit contacts (in setup mode), delete
#                    recordings, manage accounts, watch recordings.
#   role "viewer" -> monitoring only. Can watch recordings ONLY if the admin
#                    turned that on for them (can_view_recordings).
# The FIRST account created becomes the admin; after that only the admin can
# add more accounts.
 
def _load_accounts() -> list[dict]:
    try:
        data = json.loads(ACCOUNTS_PATH.read_text())
        return data if isinstance(data, list) else []
    except (FileNotFoundError, ValueError, OSError):
        pass
    # One-time migration: an older single-account file becomes the admin.
    try:
        old = json.loads(ACCOUNT_PATH.read_text())
        if isinstance(old, dict) and old.get("email"):
            migrated = [{
                "id": 1, "name": old.get("name", "Admin"), "email": old["email"],
                "salt": old.get("salt", ""), "hash": old.get("hash", ""),
                "role": "admin", "can_view_recordings": True,
            }]
            _save_accounts(migrated)
            return migrated
    except (FileNotFoundError, ValueError, OSError):
        pass
    return []
 
 
def _save_accounts(accounts: list[dict]) -> None:
    _ensure_dir()
    ACCOUNTS_PATH.write_text(json.dumps(accounts, indent=2))
    try:
        os.chmod(ACCOUNTS_PATH, 0o600)          # password hashes live here
    except OSError:
        pass
 
 
def _find_by_email(email: str):
    email = (email or "").strip().lower()
    for a in _load_accounts():
        if a.get("email") == email:
            return a
    return None
 
 
def _find_by_id(account_id):
    for a in _load_accounts():
        if a.get("id") == account_id:
            return a
    return None
 
 
def _public_account(a: dict) -> dict:
    """An account without its secrets, for sending to the browser."""
    return {
        "id": a.get("id"), "name": a.get("name"), "email": a.get("email"),
        "role": a.get("role", "viewer"),
        "can_view_recordings": bool(a.get("can_view_recordings")),
    }
 
 
def _try_add_contact(name: str, phone_raw: str) -> str:
    """Add someone to the emergency contacts too (used by the account form).
    Respects setup mode and the 5-contact cap. Returns a message to show."""
    phone = emergency_calls._normalize_phone(phone_raw)
    if not phone:
        return "Account created, but no valid phone number, so they weren't added as an emergency contact."
    if not _editing_allowed():
        return "Account created. To also add them as an emergency contact, enter setup mode on the device first."
    contacts = _load_contacts()
    if len(contacts) >= MAX_CONTACTS:
        return f"Account created, but you already have {MAX_CONTACTS} emergency contacts (the max)."
    next_id = max([c.get("id", 0) for c in contacts], default=0) + 1
    order = max([c.get("priority", 0) for c in contacts], default=0) + 1
    contacts.append({"id": next_id, "name": name.strip(), "phone": phone,
                     "priority": min(MAX_CONTACTS, order)})
    _save_contacts(contacts)
    return f"Also added as emergency contact (call #{min(MAX_CONTACTS, order)})."
 
 
def _create_account(name, email, password, role="viewer", can_view_recordings=False) -> dict:
    accounts = _load_accounts()
    salt = secrets.token_hex(16)
    new_id = max([a.get("id", 0) for a in accounts], default=0) + 1
    account = {
        "id": new_id, "name": name.strip(), "email": email.strip().lower(),
        "salt": salt, "hash": _hash_password(password, salt),
        "role": role, "can_view_recordings": bool(can_view_recordings),
    }
    accounts.append(account)
    _save_accounts(accounts)
    return account
 
 
def _new_session(account_id: int) -> str:
    token = secrets.token_urlsafe(24)
    SESSIONS[token] = account_id
    return token
 
 
# ============================================================
# The web page (dark/teal IRIS dashboard)
# ============================================================
 
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IRIS Family Dashboard</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap">
<style>
  :root{
    --bg:#1a1a1a;--surface:#232323;--surface2:#2b2b2b;--line:#363636;
    --accent:#1fc1b6;--fg:#fff;--muted:#a6a6a6;--faint:#6e6e6e;
    --ok:#1fc1b6;--warn:#e8c15a;--crit:#ff7a66;--radius:14px;--maxw:1080px;
    --sans:"Inter",system-ui,sans-serif;--display:"Space Grotesk","Inter",sans-serif;
    color-scheme:dark;
  }
  *{box-sizing:border-box}html,body{height:100%}
  [hidden]{display:none!important}   /* make the hidden attribute always win over id rules */
  body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.5;padding:0 16px}
  h1,h2,h3{font-family:var(--display);margin:0}
  button{font-family:inherit;cursor:pointer}input,select{font-family:inherit}
  .mono{font-variant-numeric:tabular-nums}
  #login{min-height:100%;display:flex;align-items:center;justify-content:center;padding-block:40px}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius)}
  .login-card{width:100%;max-width:400px;padding:30px 28px}
  .brand{display:flex;align-items:baseline;gap:10px}
  .brand .logo{font-family:var(--display);font-weight:700;font-size:30px;letter-spacing:.04em;color:var(--accent)}
  .brand .tag{color:var(--muted);font-size:13px}
  .setup-note{display:flex;gap:10px;background:var(--surface2);border:1px solid var(--line);border-radius:10px;padding:11px 13px;font-size:12.5px;color:var(--muted);margin:18px 0 20px}
  .setup-note b{color:var(--fg)}
  label{display:block;font-size:12.5px;color:var(--muted);margin:14px 0 6px;font-weight:500}
  .field{width:100%;background:var(--surface2);border:1px solid var(--line);color:var(--fg);border-radius:10px;padding:11px 12px;font-size:14px}
  .field:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
  .btn{background:var(--accent);color:#06201e;border:none;border-radius:10px;padding:12px 16px;font-weight:700;font-size:14px;width:100%;margin-top:20px}
  .btn:hover{background:#2ad3c7}
  .btn.secondary{background:transparent;color:var(--fg);border:1px solid var(--line);font-weight:600}
  .switch{text-align:center;margin-top:16px;font-size:13px;color:var(--muted)}
  .switch button{background:none;border:none;color:var(--accent);font-weight:600;font-size:13px}
  .err{color:var(--crit);font-size:13px;margin-top:12px;min-height:16px}
  #app{max-width:var(--maxw);margin:0 auto;padding-block:20px;min-height:100%}
  .topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:18px;flex-wrap:wrap}
  .topbar .brand .logo{font-size:22px}
  .who{display:flex;align-items:center;gap:12px;font-size:13px;color:var(--muted)}
  .icon-btn{background:none;border:1px solid var(--line);color:var(--muted);border-radius:8px;padding:6px 9px;font-size:12px;font-weight:600}
  .icon-btn:hover{color:var(--fg);border-color:var(--faint)}
  .icon-btn.danger:hover{color:var(--crit);border-color:var(--crit)}
  nav.tabs{display:flex;gap:6px;overflow-x:auto;border-bottom:1px solid var(--line);margin-bottom:20px}
  nav.tabs button{background:none;border:none;color:var(--muted);font-size:14px;font-weight:600;padding:11px 14px;white-space:nowrap;border-bottom:2px solid transparent}
  nav.tabs button[aria-current="true"]{color:var(--fg);border-bottom-color:var(--accent)}
  .h-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px;flex-wrap:wrap}
  .h-row h2{font-size:20px}.sub{color:var(--muted);font-size:13px;margin-top:2px}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .tile{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:15px 16px}
  .tile .k{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600}
  .tile .v{font-family:var(--display);font-size:26px;font-weight:600;margin-top:6px}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}
  .tile{container-type:inline-size}
  .tile .v{font-size:clamp(16px,15cqi,26px)}   /* shrink on narrow tiles so "Last known" stays on one line */
  .tile .s{font-size:12.5px;color:var(--muted);margin-top:6px;min-height:19px}
  .tile .s b{color:var(--fg);font-weight:600}
  .stale{opacity:.55}                 /* last-known values while the device is offline */
  .meter{height:6px;border-radius:3px;margin-top:10px;overflow:hidden;background:rgba(31,193,182,.2)}
  .meter span{display:block;height:100%;width:0;border-radius:3px;background:var(--ok);transition:width .4s}
  .meter.low{background:rgba(232,193,90,.2)}.meter.low span{background:var(--warn)}
  .meter.critical{background:rgba(255,122,102,.2)}.meter.critical span{background:var(--crit)}
  .cards2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-top:16px}
  .cards2 .block{margin-top:0}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:10px 18px;margin:0;font-size:14px}
  .kv dt{color:var(--muted)}.kv dd{margin:0;text-align:right}
  .coords{font-family:var(--display);font-size:21px;font-weight:600;margin:0 0 12px;overflow-wrap:anywhere}
  .note{color:var(--muted);font-size:13.5px}
  a.icon-btn{display:inline-block;text-decoration:none;margin-top:14px}
  .list{display:flex;flex-direction:column;gap:10px}
  .contact{display:flex;align-items:center;gap:13px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
  .contact .avatar{width:38px;height:38px;border-radius:50%;background:var(--surface2);color:var(--accent);display:flex;align-items:center;justify-content:center;font-family:var(--display);font-weight:700;flex-shrink:0}
  .contact .info{flex:1;min-width:0}.contact .nm{font-weight:600}.contact .ph{color:var(--muted);font-size:13px}
  .prio{font-size:10.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;border-radius:6px;padding:3px 7px;flex-shrink:0}
  .prio.p1{color:var(--accent);background:rgba(31,193,182,.14)}
  .prio.p2{color:var(--warn);background:rgba(232,193,90,.14)}
  .prio.p3{color:var(--faint);background:#00000030}
  .block{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:16px 17px;margin-top:16px}
  .block h3{font-size:14px;margin-bottom:12px}
  .form-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.form-grid .full{grid-column:1/-1}
  @media(max-width:480px){.form-grid{grid-template-columns:1fr}}
  .empty{color:var(--muted);font-size:14px;text-align:center;padding:30px;border:1px dashed var(--line);border-radius:12px}
  .foot-note{color:var(--faint);font-size:12px;margin-top:26px;text-align:center}
</style>
</head>
<body>
 
<section id="login">
  <div class="card login-card">
    <div class="brand"><span class="logo">IRIS</span><span class="tag">Family Dashboard</span></div>
    <div class="setup-note"><span>&#128274;</span>
      <span>This page opens when the user holds <b>Camera + Talk</b> on the IRIS device to enter <b>setup mode</b>. First time: use <b>Create account</b>. After that: <b>Sign in</b>.</span></div>
    <label for="nm">Your name <span style="color:var(--faint)">(only for Create account)</span></label>
    <input class="field" id="nm" placeholder="e.g. Sarah" autocomplete="off">
    <label for="em">Email</label>
    <input class="field" id="em" type="email" placeholder="you@example.com" autocomplete="off">
    <label for="pw">Password</label>
    <input class="field" id="pw" type="password" placeholder="At least 6 characters" autocomplete="off">
    <div style="display:flex;gap:10px;margin-top:20px">
      <button class="btn" id="signinBtn" type="button" style="margin:0">Sign in</button>
      <button class="btn secondary" id="signupBtn" type="button" style="margin:0">Create account</button>
    </div>
    <div class="err" id="loginErr"></div>
  </div>
</section>
 
<section id="app" hidden>
  <div class="topbar">
    <div class="brand"><span class="logo">IRIS</span><span class="tag">Family Dashboard</span></div>
    <div class="who"><span id="whoName">Signed in</span>
      <button class="icon-btn" id="logoutBtn">Log out</button></div>
  </div>
  <nav class="tabs" id="tabs">
    <button data-tab="overview" aria-current="true">Overview</button>
    <button data-tab="contacts">Emergency Contacts</button>
    <button data-tab="recordings" id="tabRecordings">Recordings</button>
    <button data-tab="accounts" id="tabAccounts" hidden>Accounts</button>
  </nav>
 
  <div class="panel" id="tab-overview">
    <div class="h-row"><div><h2>Overview</h2><div class="sub">Live from the device &middot; refreshes every 2 seconds.</div></div></div>
    <div class="tiles">
      <div class="tile"><div class="k">Device</div><div class="v" id="tDevice"><span class="dot" style="background:var(--faint)"></span>&hellip;</div>
        <div class="s" id="tDeviceSub"></div></div>
      <div class="tile"><div class="k">Battery</div><div class="v" id="tBattery">&mdash;</div>
        <div class="meter" id="tBatteryMeter" role="meter" aria-label="Battery charge" aria-valuemin="0" aria-valuemax="100" hidden><span></span></div>
        <div class="s" id="tBatterySub"></div></div>
      <div class="tile"><div class="k">Location</div><div class="v" id="tLocation">&mdash;</div>
        <div class="s" id="tLocationSub"></div></div>
      <div class="tile"><div class="k">Camera</div><div class="v" id="tCamera"><span class="dot" style="background:var(--faint)"></span>&hellip;</div></div>
      <div class="tile"><div class="k">Recording</div><div class="v" id="tRecording"><span class="dot" style="background:var(--faint)"></span>&hellip;</div></div>
    </div>
    <div class="cards2">
      <div class="block"><h3>Location</h3>
        <div class="note" id="locNote">&hellip;</div>
        <div id="locFix" hidden>
          <div class="coords" id="locCoords"></div>
          <dl class="kv">
            <dt>Status</dt><dd id="locState"></dd>
            <dt>Accuracy</dt><dd id="locAcc"></dd>
            <dt>Updated</dt><dd id="locAge"></dd>
          </dl>
          <a class="icon-btn" id="locMap" href="#" target="_blank" rel="noopener noreferrer">Open in Google Maps &#8599;</a>
        </div>
      </div>
      <div class="block"><h3>Device health</h3>
        <dl class="kv" id="healthList">
          <dt>Temperature</dt><dd id="hTemp">&mdash;</dd>
          <dt>Power supply</dt><dd id="hPower">&mdash;</dd>
          <dt>Battery</dt><dd id="hBatt">&mdash;</dd>
          <dt>Processor load</dt><dd id="hLoad">&mdash;</dd>
          <dt>Memory used</dt><dd id="hMem">&mdash;</dd>
          <dt>Powered on for</dt><dd id="hUp">&mdash;</dd>
          <dt>Last update</dt><dd id="hSeen">&mdash;</dd>
        </dl>
      </div>
    </div>
    <div class="block"><h3>How this connects to the device</h3>
      <div class="sub">Contacts you add on the next tab are saved straight to the device's
      <span class="mono">contacts.json</span>. When the emergency button is held, IRIS calls them in priority order.</div></div>
  </div>
 
  <div class="panel" id="tab-contacts" hidden>
    <div class="h-row">
      <div><h2>Emergency Contacts</h2><div class="sub">Called in order if the user needs help. Priority 1 is tried first.</div></div>
      <button class="icon-btn" id="addBtn">+ Add contact</button>
    </div>
    <div class="setup-note" id="editLock" hidden><span>&#128274;</span>
      <span>Editing is locked. On the device, hold <b>Camera + Talk</b> to enter setup mode, then you can add or remove contacts.</span></div>
    <div class="list" id="contactList"></div>
    <div class="block" id="contactForm" hidden>
      <h3 id="formTitle">Add contact</h3>
      <div class="form-grid">
        <div><label for="cName">Name</label><input class="field" id="cName" placeholder="e.g. Mom"></div>
        <div><label for="cPhone">Phone number</label><input class="field" id="cPhone" placeholder="+1 312 555 0101"></div>
        <div class="full"><label for="cPrio">Call order (1 = called first, up to 5 contacts)</label>
          <input class="field" id="cPrio" type="number" min="1" max="5" value="1"></div>
      </div>
      <div class="err" id="formErr"></div>
      <div style="display:flex;gap:10px;margin-top:16px">
        <button class="btn" style="width:auto;margin:0" id="saveContact">Save</button>
        <button class="btn secondary" style="width:auto;margin:0" id="cancelContact">Cancel</button>
      </div>
    </div>
    <div class="foot-note">Saved to the device's contacts.json &mdash; the emergency feature reads the same file.</div>
  </div>
 
  <div class="panel" id="tab-recordings" hidden>
    <div class="h-row"><div><h2>Recordings</h2><div class="sub">Clips the device saved &mdash; like a dashcam for incidents.</div></div>
      <button class="icon-btn" id="refreshClips">Refresh</button></div>
    <div class="list" id="clipList"></div>
    <div class="foot-note">Press the record button on the device to start/stop a clip. Saved to the device's recordings folder.</div>
  </div>
 
  <div class="panel" id="tab-accounts" hidden>
    <div class="h-row">
      <div><h2>Accounts</h2><div class="sub">You're the admin. Add family members and choose who can watch recordings.</div></div>
      <button class="icon-btn" id="addAcctBtn">+ Add person</button>
    </div>
    <div class="list" id="acctList"></div>
    <div class="block" id="acctForm" hidden>
      <h3>Add a family member</h3>
      <div class="form-grid">
        <div><label for="aName">Name</label><input class="field" id="aName" placeholder="e.g. Dad"></div>
        <div><label for="aEmail">Email</label><input class="field" id="aEmail" type="email" placeholder="dad@example.com"></div>
        <div class="full"><label for="aPass">Starter password (they can use it to sign in)</label>
          <input class="field" id="aPass" type="password" placeholder="At least 6 characters"></div>
        <div class="full"><label style="display:flex;align-items:center;gap:8px;cursor:pointer">
          <input type="checkbox" id="aRec" style="width:auto"> Allow this person to watch recordings</label></div>
        <div class="full"><label style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
          <input type="checkbox" id="aContact" style="width:auto"> Also add them as an emergency contact
          <span id="aContactNote" style="color:var(--warn);font-size:11px" hidden>&mdash; enter setup mode on the device to enable</span></label></div>
        <div class="full" id="aPhoneWrap" hidden><label for="aPhone">Phone number (for emergency calls)</label>
          <input class="field" id="aPhone" placeholder="+1 312 555 0101"></div>
      </div>
      <div class="err" id="acctErr"></div>
      <div style="display:flex;gap:10px;margin-top:16px">
        <button class="btn" style="width:auto;margin:0" id="saveAcct">Add</button>
        <button class="btn secondary" style="width:auto;margin:0" id="cancelAcct">Cancel</button>
      </div>
    </div>
    <div class="foot-note">Viewers can only monitor. Only you (admin) can edit contacts, delete recordings, and manage accounts.</div>
  </div>
</section>
 
<script>
(function(){
  "use strict";
  var $=function(s){return document.querySelector(s);};
  var token=null, signupMode=false, editingId=null, me=null;
  function isAdmin(){ return me && me.role==="admin"; }
  function canRec(){ return me && (me.role==="admin" || me.can_view_recordings); }
  function esc(s){return String(s).replace(/[&<>"]/g,function(m){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[m];});}
  function initials(n){return (n||"?").trim().charAt(0).toUpperCase();}
 
  async function api(path, method, body){
    var opts={method:method,headers:{}};
    if(token) opts.headers["X-Iris-Token"]=token;
    if(body){opts.headers["Content-Type"]="application/json";opts.body=JSON.stringify(body);}
    var r=await fetch(path,opts);
    var data={}; try{data=await r.json();}catch(e){}
    if(!r.ok) throw new Error(data.error||("Request failed ("+r.status+")"));
    return data;
  }
 
  async function doAuth(path){
    var errEl=$("#loginErr");
    errEl.style.color="var(--muted)";
    errEl.textContent="Working...";
    var email=$("#em").value.trim(), pw=$("#pw").value, name=$("#nm").value.trim();
    console.log("[IRIS] auth request", path, "email:", email);
    try{
      var res=await api(path,"POST",{name:name,email:email,password:pw});
      console.log("[IRIS] auth OK", res);
      token=res.token;
      $("#whoName").textContent=(res.name||email)+(res.role==="admin"?" (admin)":"");
      $("#login").hidden=true; $("#app").hidden=false;
      errEl.textContent="";
      await loadMe();
      loadContacts();
      startStatusPolling();
    }catch(err){
      console.log("[IRIS] auth FAILED:", err.message);
      errEl.style.color="var(--crit)";
      errEl.textContent=err.message;
    }
  }
  $("#signinBtn").addEventListener("click",function(){ doAuth("/api/login"); });
  $("#signupBtn").addEventListener("click",function(){ doAuth("/api/signup"); });
 
  $("#logoutBtn").addEventListener("click",function(){
    token=null; $("#app").hidden=true; $("#login").hidden=false;
    $("#pw").value="";
  });
 
  $("#tabs").addEventListener("click",function(e){
    var b=e.target.closest("button[data-tab]"); if(!b) return;
    document.querySelectorAll("#tabs button").forEach(function(x){x.setAttribute("aria-current",x===b);});
    $("#tab-overview").hidden=(b.dataset.tab!=="overview");
    $("#tab-contacts").hidden=(b.dataset.tab!=="contacts");
    $("#tab-recordings").hidden=(b.dataset.tab!=="recordings");
    $("#tab-accounts").hidden=(b.dataset.tab!=="accounts");
    if(b.dataset.tab==="recordings") loadRecordings();
    if(b.dataset.tab==="accounts") loadAccounts();
  });
 
  // ---- who am I + role-based UI ----
  async function loadMe(){
    try{ me=await api("/api/me","GET"); }catch(err){ me=null; }
    applyRole();
  }
  function applyRole(){
    $("#tabRecordings").hidden = !canRec();
    $("#tabAccounts").hidden = !isAdmin();
    // viewers can't add contacts at all; admin add is further gated by setup mode
    $("#addBtn").hidden = !isAdmin();
  }
 
  // ---- accounts (admin only) ----
  async function loadAccounts(){
    var el=$("#acctList");
    try{
      var res=await api("/api/accounts","GET");
      el.innerHTML=(res.accounts||[]).map(function(a){
        if(a.role==="admin"){
          return '<div class="contact"><div class="avatar">'+initials(a.name)+'</div>'
            +'<div class="info"><div class="nm">'+esc(a.name)+' <span class="prio p1">ADMIN</span></div>'
            +'<div class="ph">'+esc(a.email)+'</div></div></div>';
        }
        var checked=a.can_view_recordings?"checked":"";
        return '<div class="contact"><div class="avatar">'+initials(a.name)+'</div>'
          +'<div class="info"><div class="nm">'+esc(a.name)+'</div><div class="ph">'+esc(a.email)+'</div></div>'
          +'<label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)">'
          +'<input type="checkbox" data-recacct="'+a.id+'" '+checked+' style="width:auto"> recordings</label>'
          +'<button class="icon-btn danger" data-delacct="'+a.id+'">Remove</button></div>';
      }).join("");
    }catch(err){ el.innerHTML='<div class="empty">'+esc(err.message)+'</div>'; }
  }
  $("#addAcctBtn").addEventListener("click",function(){
    $("#aName").value="";$("#aEmail").value="";$("#aPass").value="";$("#aRec").checked=false;
    $("#aContact").checked=false;$("#aPhone").value="";$("#aPhoneWrap").hidden=true;
    $("#acctErr").textContent="";$("#acctForm").hidden=false;$("#aName").focus();
  });
  $("#aContact").addEventListener("change",function(){ $("#aPhoneWrap").hidden=!this.checked; });
  $("#cancelAcct").addEventListener("click",function(){ $("#acctForm").hidden=true; });
  $("#saveAcct").addEventListener("click",async function(){
    $("#acctErr").textContent="";
    try{
      var res=await api("/api/accounts","POST",{
        name:$("#aName").value.trim(), email:$("#aEmail").value.trim(),
        password:$("#aPass").value, can_view_recordings:$("#aRec").checked,
        add_as_contact:$("#aContact").checked, phone:$("#aPhone").value.trim() });
      $("#acctForm").hidden=true; loadAccounts();
      if(res.contact_note){ alert(res.contact_note); if($("#tab-contacts")) loadContacts(); }
    }catch(err){ $("#acctErr").textContent=err.message; }
  });
  $("#acctList").addEventListener("change",async function(e){
    var t=e.target.closest("[data-recacct]"); if(!t) return;
    try{ await api("/api/accounts/update","POST",{id:parseInt(t.dataset.recacct,10),can_view_recordings:t.checked}); }
    catch(err){ alert(err.message); loadAccounts(); }
  });
  $("#acctList").addEventListener("click",async function(e){
    var d=e.target.closest("[data-delacct]"); if(!d) return;
    if(!confirm("Remove this person's access?")) return;
    try{ await api("/api/accounts/delete","POST",{id:parseInt(d.dataset.delacct,10)}); loadAccounts(); }
    catch(err){ alert(err.message); }
  });
 
  async function loadRecordings(){
    var el=$("#clipList");
    try{
      var res=await api("/api/recordings","GET");
      var list=res.recordings||[];
      if(!list.length){ el.innerHTML='<div class="empty">No recordings yet. Say &ldquo;start recording&rdquo; to the device to make one.</div>'; return; }
      el.innerHTML=list.map(function(c){
        var name=c.file?c.file.split("/").pop():"";
        var player=name
          ? '<video controls preload="metadata" style="width:100%;border-radius:10px;margin-top:10px;background:#000" '
            +'src="/api/recordings/file?name='+encodeURIComponent(name)+'&token='+encodeURIComponent(token)+'"></video>'
          : '<div class="muted" style="font-size:12px;margin-top:8px">No video file for this clip.</div>';
        var del = (name && isAdmin())
          ? '<button class="icon-btn danger" data-delclip="'+esc(name)+'" style="margin-top:10px">Delete</button>'
          : '';
        return '<div class="block" style="margin-top:12px">'
          +'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px">'
          +'<div><div class="nm">'+esc(c.title||"Clip")+'</div>'
          +'<div class="ph">'+esc(c.when||"")+' &middot; '+esc(c.length||"")+' &middot; '+esc(c.trigger||"")+'</div></div>'
          + del + '</div>'
          +player+'</div>';
      }).join("");
    }catch(err){ el.innerHTML='<div class="empty">'+esc(err.message)+'</div>'; }
  }
  $("#refreshClips").addEventListener("click",loadRecordings);
  $("#clipList").addEventListener("click",async function(e){
    var b=e.target.closest("[data-delclip]"); if(!b) return;
    if(!confirm("Delete this recording? This can't be undone.")) return;
    try{ await api("/api/recordings/delete","POST",{name:b.dataset.delclip}); loadRecordings(); }
    catch(err){ alert(err.message); }
  });
 
  async function loadContacts(){
    try{
      var res=await api("/api/contacts","GET");
      renderContacts(res.contacts||[]);
    }catch(err){ $("#contactList").innerHTML='<div class="empty">'+esc(err.message)+'</div>'; }
  }
 
  // ---- live device status (Overview tiles + edit lock) ----
  var statusTimer=null;
  function dot(color){ return '<span class="dot" style="background:'+color+'"></span>'; }
  var LEVEL_COLOR={ok:"var(--ok)",low:"var(--warn)",critical:"var(--crit)",
                   normal:"var(--ok)",warm:"var(--warn)",hot:"var(--crit)"};
  var TEMP_WORD={normal:"Normal",warm:"Warm",hot:"Hot"};
  var lastGood=null, lastSeen=null, skew=0;   // last status with data, when (device clock), clock offset
  function num(x){ return (typeof x==="number" && isFinite(x)) ? x : null; }
  // Only touch the page when something changed (keeps text selectable while it refreshes).
  function put(el,html){ if(el._h!==html){ el._h=html; el.innerHTML=html; } }
  function txt(el,s){ put(el, esc(s)); }
  function fmtDur(min){
    min=Math.max(0,Math.round(min));
    if(min<1) return "under a minute";
    if(min<60) return min+" min";
    var h=Math.floor(min/60), m=min%60;
    if(h>=48) return Math.round(h/24)+" days";
    return h+" h"+(m?" "+m+" min":"");
  }
  function fmtAgo(sec){
    sec=Math.max(0,Math.round(sec));
    if(sec<5) return "just now";
    if(sec<60) return sec+" s ago";
    return fmtDur(sec/60)+" ago";
  }
  function clock(t){ return new Date(t*1000).toLocaleTimeString([],{hour:"numeric",minute:"2-digit"}); }

  function renderBattery(s, fresh){
    var b=s.battery_detail;
    if(!b && num(s.battery)!==null)            // older status files only had a number
      b={percent:s.battery, level:s.battery<=10?"critical":(s.battery<=20?"low":"ok"), state:"unknown"};
    var meter=$("#tBatteryMeter");
    if(!b || num(b.percent)===null){
      txt($("#tBattery"),"—"); meter.hidden=true; $("#tBattery").classList.remove("stale");
      txt($("#tBatterySub"), lastGood ? "No battery reading" : "");
      return b;
    }
    txt($("#tBattery"), b.percent+"%");
    $("#tBattery").classList.toggle("stale",!fresh);
    meter.hidden=false;
    meter.className="meter"+(b.level==="low"||b.level==="critical" ? " "+b.level : "");
    meter.firstElementChild.style.width=Math.max(0,Math.min(100,b.percent))+"%";
    meter.setAttribute("aria-valuenow",b.percent);
    if(!fresh){ txt($("#tBatterySub"),"Last known"); return b; }
    var left=num(b.minutes_left), full=num(b.minutes_to_full), tail="";
    if(b.state==="charging") tail="charging"+(full!==null?" · full in "+fmtDur(full):"");
    else if(b.state==="full") tail="fully charged";
    else if(b.state==="discharging") tail=(left!==null?"about "+fmtDur(left)+" left":"on battery");
    var lead=b.level==="critical" ? "Battery critical" : (b.level==="low" ? "Low battery" : "");
    put($("#tBatterySub"), lead ? "<b>"+lead+"</b>"+(tail?" &middot; "+esc(tail):"")
                                : esc(tail.charAt(0).toUpperCase()+tail.slice(1)));
    return b;
  }

  function renderLocation(L, fresh, now){
    var lat=L?num(L.latitude):null, lng=L?num(L.longitude):null, at=L?num(L.fix_time):null;
    var hasFix=!!L && (L.state==="live"||L.state==="last_known") && lat!==null && lng!==null;
    $("#locFix").hidden=!hasFix; $("#locNote").hidden=hasFix;
    if(!hasFix){
      var searching=!!L && L.state==="searching";
      put($("#tLocation"), searching ? dot("var(--faint)")+"Searching" : "&mdash;");
      txt($("#tLocationSub"), searching ? "Looking for satellites" : (L ? "GPS not connected" : ""));
      txt($("#locNote"), searching
        ? "Looking for GPS satellites. GPS needs a view of the sky, so indoors it can take a few minutes or not work at all."
        : (L ? "This device doesn't have a GPS connected yet." : "No location yet."));
      return;
    }
    var live=L.state==="live" && fresh, ago=at!==null ? fmtAgo(now-at) : "", acc=num(L.accuracy_m);
    put($("#tLocation"), dot(live?"var(--ok)":"var(--warn)")+(live?"Live":"Last known"));
    txt($("#tLocationSub"), "Updated "+ago+(acc!==null?" · ±"+Math.round(acc)+" m":""));
    var coords=lat.toFixed(5)+", "+lng.toFixed(5);
    txt($("#locCoords"), coords);
    $("#locMap").href="https://www.google.com/maps/search/?api=1&query="+encodeURIComponent(lat+","+lng);
    put($("#locState"), live ? dot("var(--ok)")+"Live GPS fix"
      : dot("var(--warn)")+(fresh ? "Last known &middot; GPS signal lost" : "Last known &middot; device offline"));
    txt($("#locAcc"), acc!==null ? "± "+Math.round(acc)+" m" : "—");
    txt($("#locAge"), at!==null ? ago+" · "+clock(at) : "—");
  }

  function renderHealth(d, b, fresh, now){
    d=d||{};
    var t=num(d.cpu_temp_c), p=d.power, v=b?num(b.voltage):null, r=b?num(b.rate_pct_per_hour):null;
    put($("#hTemp"), t===null ? "&mdash;"
      : dot(LEVEL_COLOR[d.temp_level]||"var(--faint)")+Math.round(t)+" &deg;C &middot; "+esc(TEMP_WORD[d.temp_level]||""));
    put($("#hPower"), !p ? "&mdash;"
      : p.under_voltage_now ? dot("var(--crit)")+"Voltage too low"
      : p.under_voltage_since_boot ? dot("var(--warn)")+"Dipped earlier" : dot("var(--ok)")+"OK");
    txt($("#hBatt"), v===null ? "—" : v.toFixed(2)+" V"
      + (r!==null && Math.abs(r)>=1 ? (r<0?" · draining ":" · charging ")+Math.abs(r).toFixed(0)+"%/h" : ""));
    txt($("#hLoad"), num(d.load_pct)!==null ? d.load_pct+"%" : "—");
    txt($("#hMem"), num(d.memory_pct)!==null ? d.memory_pct+"%" : "—");
    txt($("#hUp"), num(d.uptime_s)!==null ? fmtDur(d.uptime_s/60) : "—");
    txt($("#hSeen"), lastSeen!==null ? fmtAgo(now-lastSeen) : "—");
    ["#hTemp","#hPower","#hBatt","#hLoad","#hMem","#hUp"].forEach(function(id){ $(id).classList.toggle("stale",!fresh); });
  }

  function applyStatus(res){
    if(res && num(res.server_time)!==null) skew=res.server_time-Date.now()/1000;
    var now=Date.now()/1000+skew;                       // "now" on the device's clock
    var hasData=!!res && num(res.updated)!==null;
    if(hasData){ lastGood=res; lastSeen=res.updated; }
    var s=hasData ? res : (lastGood||{});               // offline: keep showing the last known values
    var fresh=hasData && (now-res.updated < 8);
    var up=num((s.device||{}).uptime_s);
    put($("#tDevice"), fresh ? dot("var(--ok)")+"Online" : dot("var(--faint)")+"Offline");
    txt($("#tDeviceSub"), fresh ? (up!==null ? "On for "+fmtDur(up/60) : "Connected")
      : (lastSeen!==null ? "Last seen "+fmtAgo(now-lastSeen) : (res ? "IRIS isn't sending data" : "Can't reach the device")));
    var b=renderBattery(s, fresh);
    renderLocation(s.location, fresh, now);
    renderHealth(s.device, b, fresh, now);
    put($("#tCamera"), !fresh ? "&mdash;" : s.camera_on ? dot("var(--ok)")+"On" : dot("var(--faint)")+"Off");
    put($("#tRecording"), !fresh ? "&mdash;" : s.recording ? dot("var(--crit)")+"Recording" : dot("var(--faint)")+"Idle");
    // Editing is only allowed in setup mode. If there's no app status at all, treat as editable.
    var locked = fresh ? !s.editing_allowed : false;
    $("#editLock").hidden = !(locked && isAdmin());    // only the admin edits, so only show it to them
    $("#addBtn").disabled = locked;
    $("#addBtn").style.opacity = locked ? "0.4" : "";
    $("#addBtn").style.pointerEvents = locked ? "none" : "";
    // The "also add as emergency contact" option needs setup mode: keep it
    // visible but unclickable, with a note, when editing is locked.
    var ac=$("#aContact");
    if(ac){
      ac.disabled = locked;
      ac.parentElement.style.opacity = locked ? "0.55" : "";
      $("#aContactNote").hidden = !locked;
      if(locked){ ac.checked=false; $("#aPhoneWrap").hidden=true; }
    }
  }
  async function pollStatus(){
    try{ applyStatus(await api("/api/status","GET")); }
    catch(err){ applyStatus(null); }
  }
  function startStatusPolling(){
    pollStatus();
    if(statusTimer) clearInterval(statusTimer);
    statusTimer=setInterval(pollStatus, 2000);
  }
  function renderContacts(list){
    var el=$("#contactList");
    if(!list.length){ el.innerHTML='<div class="empty">No contacts yet. Add the first person IRIS should call.</div>'; return; }
    el.innerHTML=list.map(function(x){
      var actions = isAdmin()
        ? '<button class="icon-btn" data-edit="'+x.id+'">Edit</button>'
          +'<button class="icon-btn danger" data-del="'+x.id+'">Remove</button>'
        : '';
      var pcls = x.priority===1 ? "p1" : (x.priority===2 ? "p2" : "p3");
      return '<div class="contact"><div class="avatar">'+initials(x.name)+'</div>'
        +'<div class="info"><div class="nm">'+esc(x.name)+'</div><div class="ph mono">'+esc(x.phone)+'</div></div>'
        +'<span class="prio '+pcls+'">Call #'+x.priority+'</span>'
        + actions +'</div>';
    }).join("");
    window._contacts=list;
  }
 
  function openForm(c){
    editingId=c?c.id:null;
    $("#formTitle").textContent=c?"Edit contact":"Add contact";
    $("#cName").value=c?c.name:""; $("#cPhone").value=c?c.phone:""; $("#cPrio").value=c?c.priority:"1";
    $("#formErr").textContent=""; $("#contactForm").hidden=false; $("#cName").focus();
  }
  $("#addBtn").addEventListener("click",function(){ openForm(null); });
  $("#cancelContact").addEventListener("click",function(){ $("#contactForm").hidden=true; });
  $("#saveContact").addEventListener("click",async function(){
    var name=$("#cName").value.trim(), phone=$("#cPhone").value.trim(), prio=parseInt($("#cPrio").value,10);
    $("#formErr").textContent="";
    if(!name||!phone){ $("#formErr").textContent="Name and phone are both required."; return; }
    try{
      await api("/api/contacts","POST",{id:editingId,name:name,phone:phone,priority:prio});
      $("#contactForm").hidden=true; loadContacts();
    }catch(err){ $("#formErr").textContent=err.message; }
  });
  $("#contactList").addEventListener("click",async function(e){
    var ed=e.target.closest("[data-edit]"), del=e.target.closest("[data-del]");
    if(ed){ var c=(window._contacts||[]).filter(function(x){return x.id==ed.dataset.edit;})[0]; if(c) openForm(c); }
    if(del){ try{ await api("/api/contacts/delete","POST",{id:parseInt(del.dataset.del,10)}); loadContacts(); }catch(err){} }
  });
})();
</script>
</body>
</html>
"""
 
 
# ============================================================
# HTTP handler
# ============================================================
 
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # keep the terminal quiet
        pass
 
    # -- small helpers -------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
 
    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}
 
    def _auth_account(self, token: str | None = None) -> dict | None:
        """The signed-in account for this request (or None)."""
        if token is None:
            token = self.headers.get("X-Iris-Token", "")
        account_id = SESSIONS.get(token)
        if account_id is None:
            return None
        return _find_by_id(account_id)
 
    def _is_admin(self, account: dict | None) -> bool:
        return bool(account) and account.get("role") == "admin"
 
    def _can_view_recordings(self, account: dict | None) -> bool:
        return bool(account) and (
            account.get("role") == "admin" or account.get("can_view_recordings"))
 
    # -- routes --------------------------------------------------------
    def do_GET(self):
        if self.path == "/favicon.ico":        # browsers always ask; answer quietly
            self.send_response(204)
            self.end_headers()
            return
        if self.path == "/" or self.path.startswith("/?"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)
            return
        account = self._auth_account()
        if self.path == "/api/me":
            if account is None:
                return self._send_json({"error": "Please sign in first."}, 401)
            return self._send_json(_public_account(account))
        if self.path == "/api/contacts":
            if account is None:
                return self._send_json({"error": "Please sign in first."}, 401)
            return self._send_json({"contacts": _load_contacts()})
        if self.path == "/api/status":
            if account is None:
                return self._send_json({"error": "Please sign in first."}, 401)
            status = _read_status()
            if not isinstance(status, dict):
                status = {}
            # The device's own clock, so the page can say "last seen 3 min ago"
            # correctly even if the phone's clock and the Pi's clock disagree.
            status["server_time"] = time.time()
            return self._send_json(status)
        if self.path == "/api/accounts":                 # admin: list everyone
            if not self._is_admin(account):
                return self._send_json({"error": "Admins only."}, 403)
            return self._send_json({"accounts": [_public_account(a) for a in _load_accounts()]})
        if self.path == "/api/recordings":
            if not self._can_view_recordings(account):
                return self._send_json(
                    {"error": "You don't have access to recordings."}, 403)
            try:
                clips = recording.list_recordings()
            except Exception:
                clips = []
            return self._send_json({"recordings": clips})
        if self.path.startswith("/api/recordings/file"):
            return self._serve_video()
        self._send_json({"error": "Not found"}, 404)
 
    # -- stream a recorded video so the browser can play it ------------
    def _serve_video(self):
        query = parse_qs(urlparse(self.path).query)
        # The <video> tag can't send our auth header, so the token rides in
        # the query string instead.
        token = (query.get("token") or [""])[0]
        if not self._can_view_recordings(self._auth_account(token)):
            return self._send_json({"error": "You don't have access to recordings."}, 403)
        name = os.path.basename((query.get("name") or [""])[0])
        if not name:
            return self._send_json({"error": "No file named."}, 400)
 
        video_dir = recording.VIDEO_DIR.resolve()
        path = (video_dir / name).resolve()
        # Only serve files that really live inside the recordings folder.
        if video_dir not in path.parents or not path.is_file():
            return self._send_json({"error": "Recording not found."}, 404)
 
        size = path.stat().st_size
        rng = self.headers.get("Range", "")
        start, end = 0, size - 1
        partial = False
        if rng.startswith("bytes="):
            partial = True
            s, _, e = rng[len("bytes="):].partition("-")
            try:
                start = int(s) if s else 0
                end = int(e) if e else size - 1
            except ValueError:
                start, end = 0, size - 1
            start = max(0, start)
            end = min(end, size - 1)
 
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass        # the browser closed the video stream; that's fine
 
    def do_POST(self):
        data = self._read_body()
 
        if self.path == "/api/signup":
            # Only the FIRST account can be made this way, and it's the admin.
            name = str(data.get("name", "")).strip()
            email = str(data.get("email", "")).strip()
            password = str(data.get("password", ""))
            if not name or not email or len(password) < 6:
                return self._send_json(
                    {"error": "Enter a name, email, and a password of 6+ characters."}, 400)
            if _load_accounts():
                return self._send_json(
                    {"error": "This device already has an admin. Ask them to add you."}, 400)
            account = _create_account(name, email, password,
                                      role="admin", can_view_recordings=True)
            return self._send_json({"token": _new_session(account["id"]),
                                    "name": account["name"], "role": "admin"})
 
        if self.path == "/api/login":
            email = str(data.get("email", "")).strip().lower()
            password = str(data.get("password", ""))
            account = _find_by_email(email)
            if account is None:
                return self._send_json({"error": "Wrong email or password."}, 401)
            ok = _hash_password(password, account.get("salt", "")) == account.get("hash")
            if not ok:
                return self._send_json({"error": "Wrong email or password."}, 401)
            return self._send_json({"token": _new_session(account["id"]),
                                    "name": account["name"], "role": account.get("role")})
 
        # everything below needs a signed-in session
        account = self._auth_account()
        if account is None:
            return self._send_json({"error": "Please sign in first."}, 401)
 
        # --- account management (admin only) -----------------------------
        if self.path in ("/api/accounts", "/api/accounts/update", "/api/accounts/delete"):
            if not self._is_admin(account):
                return self._send_json({"error": "Admins only."}, 403)
            return self._handle_accounts(self.path, data, account)
 
        # Editing or deleting contacts: admin only, AND only in setup mode.
        if self.path in ("/api/contacts", "/api/contacts/delete"):
            if not self._is_admin(account):
                return self._send_json({"error": "Only the admin can change contacts."}, 403)
            if not _editing_allowed():
                return self._send_json(
                    {"error": "Editing is locked. On the device, hold Camera + Talk "
                              "to enter setup mode, then edit contacts."}, 403)
 
        # Deleting recordings: admin only.
        if self.path == "/api/recordings/delete" and not self._is_admin(account):
            return self._send_json({"error": "Only the admin can delete recordings."}, 403)
 
        if self.path == "/api/contacts":
            name = str(data.get("name", "")).strip()
            phone = emergency_calls._normalize_phone(data.get("phone", ""))
            if not name or not phone:
                return self._send_json(
                    {"error": "Enter a name and a valid phone number (e.g. +13125550101)."}, 400)
            try:
                priority = int(data.get("priority", 1))
            except (TypeError, ValueError):
                priority = 1
            priority = max(1, min(MAX_CONTACTS, priority))   # call order, 1..5
 
            contacts = _load_contacts()
            cid = data.get("id")
            if cid:                                  # edit existing
                for c in contacts:
                    if c["id"] == cid:
                        c.update(name=name, phone=phone, priority=priority)
                        break
                else:
                    cid = None
            if not cid:                              # add new
                if len(contacts) >= MAX_CONTACTS:
                    return self._send_json(
                        {"error": f"You can have at most {MAX_CONTACTS} emergency contacts."}, 400)
                next_id = max([c["id"] for c in contacts], default=0) + 1
                contacts.append({"id": next_id, "name": name, "phone": phone, "priority": priority})
            _save_contacts(contacts)
            return self._send_json({"ok": True, "contacts": _load_contacts()})
 
        if self.path == "/api/contacts/delete":
            cid = data.get("id")
            contacts = [c for c in _load_contacts() if c["id"] != cid]
            _save_contacts(contacts)
            return self._send_json({"ok": True, "contacts": contacts})
 
        if self.path == "/api/recordings/delete":
            result = recording.delete_recording(str(data.get("name", "")))
            status = 200 if result.get("status") == "deleted" else 400
            return self._send_json(result, status)
 
        self._send_json({"error": "Not found"}, 404)
 
    # -- admin: add / change / remove family accounts ------------------
    def _handle_accounts(self, path, data, admin):
        if path == "/api/accounts":                      # create a viewer
            name = str(data.get("name", "")).strip()
            email = str(data.get("email", "")).strip()
            password = str(data.get("password", ""))
            if not name or not email or len(password) < 6:
                return self._send_json(
                    {"error": "Enter a name, email, and a password of 6+ characters."}, 400)
            if _find_by_email(email) is not None:
                return self._send_json({"error": "An account with that email already exists."}, 400)
            acct = _create_account(name, email, password, role="viewer",
                                   can_view_recordings=bool(data.get("can_view_recordings")))
            result = {"ok": True, "account": _public_account(acct)}
            # Optional: also add this person to the emergency contacts list.
            if data.get("add_as_contact"):
                result["contact_note"] = _try_add_contact(name, data.get("phone", ""))
            return self._send_json(result)
 
        if path == "/api/accounts/update":               # toggle recordings access
            aid = data.get("id")
            accounts = _load_accounts()
            for a in accounts:
                if a.get("id") == aid:
                    if a.get("role") == "admin":
                        return self._send_json({"error": "The admin's access can't be changed."}, 400)
                    if "can_view_recordings" in data:
                        a["can_view_recordings"] = bool(data["can_view_recordings"])
                    _save_accounts(accounts)
                    return self._send_json({"ok": True, "account": _public_account(a)})
            return self._send_json({"error": "Account not found."}, 404)
 
        if path == "/api/accounts/delete":               # remove a viewer
            aid = data.get("id")
            if aid == admin.get("id"):
                return self._send_json({"error": "You can't delete your own admin account."}, 400)
            target = _find_by_id(aid)
            if target and target.get("role") == "admin":
                return self._send_json({"error": "The admin account can't be deleted."}, 400)
            _save_accounts([a for a in _load_accounts() if a.get("id") != aid])
            for tok, sid in list(SESSIONS.items()):      # log that person out
                if sid == aid:
                    del SESSIONS[tok]
            return self._send_json({"ok": True})
 
        return self._send_json({"error": "Not found"}, 404)
 
 
def _lan_ip() -> str:
    """This machine's address on the local network (for same-WiFi access)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))     # no data sent; just picks the WiFi interface
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()
 
 
def main():
    _ensure_dir()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    lan = _lan_ip()
    print("=" * 60)
    print("  IRIS setup website is running.")
    print(f"  On THIS device:      http://localhost:{PORT}")
    print(f"  On the same WiFi:    http://{lan}:{PORT}")
    print(f"  Contacts are saved to: {CONTACTS_PATH}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()
 
 
if __name__ == "__main__":
    main()
 