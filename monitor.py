import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from openai import AzureOpenAI


# ============================================================
# Configuration
# ============================================================

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

TRAINING_URL = os.getenv(
    "TRAINING_URL",
    "https://youth.europa.eu/solidarity/dashboard/training-humanitarian-aid_en",
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PROFILE_DIR = Path("playwright_profile")
STATE_FILE = Path("state.json")

PAGE_TIMEOUT = 60_000


# ============================================================
# Validation
# ============================================================

def validate_config():
    missing = []

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        print(
            "Missing environment variables:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )
        sys.exit(1)


# ============================================================
# Telegram
# ============================================================

def send_telegram(message: str):
    """
    Send a Telegram notification.
    """

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": False,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    print("Telegram notification sent.")


# ============================================================
# State
# ============================================================

def load_state():
    if not STATE_FILE.exists():
        return None

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read state file: {e}")
        return None


def save_state(state):
    temp_file = STATE_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            state,
            f,
            indent=2,
            ensure_ascii=False,
        )

    temp_file.replace(STATE_FILE)


# ============================================================
# Text processing
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize whitespace so tiny formatting changes don't
    trigger a notification.
    """

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def fingerprint(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# Page classification
# ============================================================


def classify_with_ai(page_text: str, previous_status: str | None):
    prompt = f"""
You are monitoring the European Youth Portal page for
Humanitarian Aid Volunteering training.

Your job is ONLY to determine whether registration/access is
currently open.

Previous known status:
{previous_status or "NONE"}

CURRENT PAGE TEXT:
------------------
{page_text}
------------------

Classify the CURRENT page into exactly one of:

OPEN
CLOSED
UNKNOWN

Definitions:

OPEN:
The page provides clear evidence that people can currently
register for or access the Humanitarian Aid Volunteering
online training or face-to-face training.

CLOSED:
The page clearly states that registration/access is closed,
suspended, unavailable, or that registrations are not currently
being accepted.

UNKNOWN:
The page does not provide enough reliable evidence to determine
the status, or the page is a login page, error page, CAPTCHA,
server error, incomplete page, unrelated page, etc.

Important:
- Do NOT infer that registration is open merely because the page
  contains words such as "open", "available", "opportunities",
  "activities", etc.
- Look specifically at Humanitarian Aid Volunteering training
  registration/access.
- Do not use the previous status as evidence for the current status.
- Do not guess.
- If there is ambiguity, return UNKNOWN.
- Base your decision only on the supplied page text.

Return ONLY valid JSON:

{{
  "status": "OPEN | CLOSED | UNKNOWN",
  "confidence": 0.0,
  "reason": "short explanation",
  "evidence": "short exact relevant excerpt"
}}
"""

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict webpage-status classifier. "
                    "Never guess. Ambiguous cases are UNKNOWN."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        response_format={
            "type": "json_object"
        },
    )

    result = json.loads(
        response.choices[0].message.content
    )

    return result


# ============================================================
# Relevant content extraction
# ============================================================

def extract_relevant_text(page):
    """
    Try to isolate the actual training page text.

    We intentionally use the whole body as a fallback because
    the site's DOM can change.
    """

    try:
        body_text = page.locator("body").inner_text(
            timeout=10_000
        )
    except Exception:
        return ""

    normalized = normalize_text(body_text)

    # We care especially about text around:
    # Humanitarian Aid
    # registration
    # online training
    # face-to-face training

    keywords = [
        "humanitarian aid",
        "online training",
        "face-to-face",
        "registration",
        "volunteering",
    ]

    if not any(
        keyword.lower() in normalized.lower()
        for keyword in keywords
    ):
        return normalized

    # Keep the complete page text for fingerprinting.
    # This avoids accidentally missing a changed sentence.
    return normalized


# ============================================================
# Browser
# ============================================================

def launch_browser(playwright):
    """
    Persistent Chromium profile.

    The important part is user_data_dir.
    Cookies/local storage/login state remain there.
    """

    PROFILE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    browser = playwright.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR.absolute()),
        headless=True,

        # Useful for sites that behave differently depending
        # on browser automation flags.
        args=[
            "--disable-blink-features=AutomationControlled",
        ],

        viewport={
            "width": 1440,
            "height": 1000,
        },
    )

    return browser


# ============================================================
# Login mode
# ============================================================

def login_mode():
    print()
    print("=" * 70)
    print("LOGIN MODE")
    print("=" * 70)
    print()
    print("A browser will open.")
    print()
    print("Log in to the European Youth Portal normally.")
    print()
    print(
        "When you can access the Humanitarian Aid training page, "
        "come back to this terminal."
    )
    print()

    with sync_playwright() as p:

        PROFILE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR.absolute()),
            headless=False,
            viewport={
                "width": 1440,
                "height": 1000,
            },
        )

        page = browser.new_page()

        try:
            page.goto(
                TRAINING_URL,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT,
            )
        except Exception as e:
            print(f"Initial navigation error: {e}")

        print()
        print("Browser is open.")
        print()
        input(
            "Press ENTER here after you have finished logging in..."
        )

        try:
            page.goto(
                TRAINING_URL,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT,
            )

            time.sleep(3)

            text = page.locator("body").inner_text(
                timeout=10_000
            )

            ai_result = classify_with_ai(text)

            status = ai_result["status"]

            print("AI status:", status)
            print("Confidence:", ai_result["confidence"])
            print("Reason:", ai_result["reason"])
            print("Evidence:", ai_result["evidence"])
            print()
            print("Page status:", status)
            print()

            if status == "UNKNOWN":
                print(
                    "WARNING: I could not confidently identify "
                    "the training page status."
                )
                print(
                    "The saved browser session may still be valid."
                )
            else:
                print(
                    "Login/session appears usable."
                )

        except Exception as e:
            print(
                f"Could not verify page: {e}"
            )

        print()
        print(
            "Your browser session has been saved to:"
        )
        print(
            PROFILE_DIR.absolute()
        )
        print()

        input(
            "Press ENTER to close the browser..."
        )

        browser.close()


# ============================================================
# Monitoring
# ============================================================

def monitor():
    validate_config()

    previous_state = load_state()

    print()
    print("=" * 70)
    print("Humanitarian Aid Training Monitor")
    print("=" * 70)
    print()

    print(
        "URL:",
        TRAINING_URL,
    )

    with sync_playwright() as p:

        browser = launch_browser(p)

        try:

            page = browser.pages[0]

            if not page:
                page = browser.new_page()

            print()
            print("Opening page...")

            try:
                response = page.goto(
                    TRAINING_URL,
                    wait_until="domcontentloaded",
                    timeout=PAGE_TIMEOUT,
                )

                if response:
                    print(
                        "HTTP status:",
                        response.status,
                    )

                    # Don't classify HTTP failures as CLOSED.
                    if response.status >= 400:
                        print(
                            "HTTP error. Treating status as UNKNOWN."
                        )

                        current_state = {
                            "status": "UNKNOWN",
                            "fingerprint": None,
                            "timestamp": datetime.now().isoformat(),
                            "reason": f"HTTP {response.status}",
                        }

                        handle_result(
                            previous_state,
                            current_state,
                        )

                        return

            except PlaywrightTimeoutError:

                print(
                    "Page load timed out."
                )

                current_state = {
                    "status": "UNKNOWN",
                    "fingerprint": None,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "Page timeout",
                }

                handle_result(
                    previous_state,
                    current_state,
                )

                return

            except Exception as e:

                print(
                    f"Navigation failed: {e}"
                )

                current_state = {
                    "status": "UNKNOWN",
                    "fingerprint": None,
                    "timestamp": datetime.now().isoformat(),
                    "reason": str(e),
                }

                handle_result(
                    previous_state,
                    current_state,
                )

                return

            # Give JavaScript a little time to finish rendering.
            time.sleep(3)

            # ------------------------------------------------
            # Detect login/authentication
            # ------------------------------------------------

            current_url = page.url

            print(
                "Final URL:",
                current_url,
            )

            # If we were redirected somewhere obviously related
            # to authentication, don't classify it.
            auth_url_patterns = [
                "/login",
                "/signin",
                "/sign-in",
                "/authenticate",
            ]

            if any(
                x in current_url.lower()
                for x in auth_url_patterns
            ):
                current_state = {
                    "status": "UNKNOWN",
                    "fingerprint": None,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "Authentication page",
                }

                handle_result(
                    previous_state,
                    current_state,
                )

                return

            # ------------------------------------------------
            # Read page
            # ------------------------------------------------

            text = extract_relevant_text(page)

            if not text:

                current_state = {
                    "status": "UNKNOWN",
                    "fingerprint": None,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "Empty page",
                }

                handle_result(
                    previous_state,
                    current_state,
                )

                return

            # ------------------------------------------------
            # Classify
            # ------------------------------------------------

            ai_result = classify_with_ai(
                    text,
                    previous_state.get("status") if previous_state else None,
                )

            if (
                ai_result["status"] == "OPEN"
                and ai_result["confidence"] >= 0.90
            ):
                status = "OPEN"

            elif (
                ai_result["status"] == "CLOSED"
                and ai_result["confidence"] >= 0.90
            ):
                status = "CLOSED"

            else:
                status = "UNKNOWN"

            print("AI status:", status)
            print("Confidence:", ai_result["confidence"])
            print("Reason:", ai_result["reason"])
            print("Evidence:", ai_result["evidence"])

            page_hash = fingerprint(text)

            current_state = {
                "status": status,
                "fingerprint": page_hash,
                "timestamp": datetime.now().isoformat(),
                "url": current_url,
                "reason": None,
            }

            print()
            print("Detected status:", status)
            print("Fingerprint:", page_hash[:16])
            print()

            # ------------------------------------------------
            # Handle result
            # ------------------------------------------------

            handle_result(
                previous_state,
                current_state,
                page_text=text,
            )

        finally:
            browser.close()


# ============================================================
# Result handling
# ============================================================

def handle_result(
    previous_state,
    current_state,
    page_text=None,
):
    current_status = current_state["status"]

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if current_status == "UNKNOWN":

        print(
            "Result is UNKNOWN."
        )

        print(
            "No notification will be sent."
        )

        # Don't overwrite a known good state with UNKNOWN.
        if previous_state:
            print(
                "Keeping previous known state:",
                previous_state.get("status"),
            )

        else:
            save_state(current_state)

        return

    # --------------------------------------------------------
    # First successful run
    # --------------------------------------------------------

    if previous_state is None:

        print(
            "First successful run."
        )

        print(
            "Saving baseline without notification."
        )

        save_state(current_state)

        return

    previous_status = previous_state.get(
        "status"
    )

    previous_hash = previous_state.get(
        "fingerprint"
    )

    current_hash = current_state.get(
        "fingerprint"
    )

    print(
        "Previous status:",
        previous_status,
    )

    # --------------------------------------------------------
    # CLOSED -> OPEN
    # --------------------------------------------------------

    if (
        previous_status == "CLOSED"
        and current_status == "OPEN"
    ):

        message = (
            "🚨 Humanitarian Aid Training Update\n\n"
            "Registration/access appears to be OPEN.\n\n"
            "The European Youth Portal page has changed "
            "from the previous CLOSED state.\n\n"
            f"{TRAINING_URL}"
        )

        send_telegram(message)

        save_state(current_state)

        return

    # --------------------------------------------------------
    # OPEN remains OPEN
    # --------------------------------------------------------

    if (
        previous_status == "OPEN"
        and current_status == "OPEN"
    ):

        if previous_hash != current_hash:

            message = (
                "🔔 Humanitarian Aid Training Update\n\n"
                "The Humanitarian Aid training page has "
                "changed while registration/access appears OPEN.\n\n"
                f"{TRAINING_URL}"
            )

            send_telegram(message)

        else:

            print(
                "Still OPEN. No change."
            )

        save_state(current_state)

        return

    # --------------------------------------------------------
    # CLOSED remains CLOSED
    # --------------------------------------------------------

    if (
        previous_status == "CLOSED"
        and current_status == "CLOSED"
    ):

        print(
            "Still CLOSED."
        )

        # Ignore harmless page changes while the explicit
        # closure notice remains present.
        if previous_hash != current_hash:
            print(
                "Page fingerprint changed, but status "
                "remains CLOSED. No notification."
            )

        save_state(current_state)

        return

    # --------------------------------------------------------
    # UNKNOWN -> known
    # --------------------------------------------------------

    if (
        previous_status == "UNKNOWN"
        and current_status == "OPEN"
    ):

        message = (
            "🚨 Humanitarian Aid Training Update\n\n"
            "The training page is now readable and "
            "registration/access appears OPEN.\n\n"
            f"{TRAINING_URL}"
        )

        send_telegram(message)

        save_state(current_state)

        return

    if (
        previous_status == "UNKNOWN"
        and current_status == "CLOSED"
    ):

        print(
            "Page is readable again and remains CLOSED."
        )

        save_state(current_state)

        return

    # --------------------------------------------------------
    # Any other transition
    # --------------------------------------------------------

    if previous_status != current_status:

        message = (
            "🔔 Humanitarian Aid Training Update\n\n"
            f"Previous status: {previous_status}\n"
            f"Current status: {current_status}\n\n"
            f"{TRAINING_URL}"
        )

        send_telegram(message)

    save_state(current_state)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--login",
        action="store_true",
        help="Open a visible browser so you can log in.",
    )

    parser.add_argument(
        "--test-telegram",
        action="store_true",
        help="Send a test Telegram notification.",
    )

    args = parser.parse_args()

    if args.test_telegram:

        validate_config()

        send_telegram(
            "✅ Humanitarian Aid monitor is connected correctly."
        )

        return

    if args.login:

        login_mode()

        return

    monitor()


if __name__ == "__main__":
    main()