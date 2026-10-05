import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


# ============================================================
# Configuration
# ============================================================

load_dotenv()

TRAINING_URL = os.getenv(
    "TRAINING_URL",
    "https://youth.europa.eu/solidarity/dashboard/training-humanitarian-aid_en",
)

AUTH_FILE = Path("auth_state.json")

PAGE_TIMEOUT = 60_000


# ============================================================
# Environment variables
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


# ============================================================
# Azure OpenAI
# ============================================================

client = None

if (
    AZURE_OPENAI_ENDPOINT
    and AZURE_OPENAI_API_KEY
    and AZURE_OPENAI_API_VERSION
):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


# ============================================================
# Validation
# ============================================================

def validate_config():
    missing = []

    required_variables = {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
    }

    for name, value in required_variables.items():
        if not value:
            missing.append(name)

    if not AUTH_FILE.exists():
        print(
            f"Missing authentication file: "
            f"{AUTH_FILE.absolute()}"
        )
        print()
        print(
            "Make sure auth_state.json exists."
        )
        sys.exit(1)

    if missing:
        print()
        print("Missing environment variables:")
        print()

        for variable in missing:
            print(f"  - {variable}")

        print()
        sys.exit(1)


# ============================================================
# Telegram
# ============================================================

def send_telegram(message: str):
    """
    Send a Telegram message.
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
# Text processing
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize whitespace.
    """

    text = text.replace("\xa0", " ")

    text = " ".join(text.split())

    return text.strip()


# ============================================================
# Page text extraction
# ============================================================

def extract_page_text(page) -> str:
    """
    Extract all visible text from the page.
    """

    try:
        body_text = page.locator(
            "body"
        ).inner_text(
            timeout=10_000
        )

    except Exception as e:

        print(
            f"Could not read page body: {e}"
        )

        return ""

    return normalize_text(body_text)


# ============================================================
# AI classification
# ============================================================

def classify_with_ai(page_text: str):
    """
    Classify the current Humanitarian Aid training page.

    Returns:

        OPEN
        CLOSED
        UNKNOWN
    """

    prompt = f"""
You are monitoring the European Youth Portal page for
Humanitarian Aid Volunteering training.

Your job is ONLY to determine whether registration/access is
currently open.

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

- Look specifically at Humanitarian Aid Volunteering
  training registration/access.

- Do NOT guess.

- If there is ambiguity, return UNKNOWN.

- Base your decision ONLY on the supplied page text.

Return ONLY valid JSON:

{{
    "status": "OPEN | CLOSED | UNKNOWN",
    "confidence": 0.0,
    "reason": "short explanation",
    "evidence": "short exact relevant excerpt"
}}
"""

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
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

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Azure OpenAI returned an empty response."
        )

    result = json.loads(content)

    # --------------------------------------------------------
    # Validate status
    # --------------------------------------------------------

    status = result.get("status")

    if status not in {
        "OPEN",
        "CLOSED",
        "UNKNOWN",
    }:
        status = "UNKNOWN"

    # --------------------------------------------------------
    # Validate confidence
    # --------------------------------------------------------

    try:
        confidence = float(
            result.get("confidence", 0)
        )

    except (TypeError, ValueError):

        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    return {
        "status": status,
        "confidence": confidence,
        "reason": str(
            result.get("reason", "")
        ),
        "evidence": str(
            result.get("evidence", "")
        ),
    }


# ============================================================
# Authentication detection
# ============================================================

def is_authentication_page(
    current_url: str,
    page_text: str,
) -> bool:
    """
    Detect obvious authentication redirects.

    Authentication failures are reported as UNKNOWN rather
    than CLOSED.
    """

    url = current_url.lower()

    auth_url_patterns = [
        "/login",
        "/signin",
        "/sign-in",
        "/authenticate",
        "/account/login",
    ]

    if any(
        pattern in url
        for pattern in auth_url_patterns
    ):
        return True

    text = page_text.lower()

    auth_phrases = [
        "log in to your account",
        "sign in to your account",
        "please log in",
        "please sign in",
    ]

    return any(
        phrase in text
        for phrase in auth_phrases
    )


# ============================================================
# Telegram message
# ============================================================

def build_telegram_message(
    status: str,
    confidence: float,
    reason: str,
    evidence: str,
    url: str,
):
    """
    Build the Telegram message containing the current state.
    """

    emoji = {
        "OPEN": "🟢",
        "CLOSED": "🔴",
        "UNKNOWN": "🟡",
    }.get(
        status,
        "⚪",
    )

    return (
        f"{emoji} Humanitarian Aid Training Monitor\n\n"
        f"Status: {status}\n"
        f"Confidence: {confidence:.0%}\n\n"
        f"Reason:\n"
        f"{reason}\n\n"
        f"Evidence:\n"
        f"{evidence}\n\n"
        f"URL:\n"
        f"{url}"
    )


# ============================================================
# Main monitor
# ============================================================

def monitor():
    """
    Run one monitoring cycle.

    Every execution sends the current state to Telegram.
    """

    validate_config()

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

        browser = None

        try:

            # ------------------------------------------------
            # Launch browser
            # ------------------------------------------------

            print()
            print(
                "Launching browser..."
            )

            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                ],
            )

            context = browser.new_context(
                storage_state=str(
                    AUTH_FILE.absolute()
                ),
                viewport={
                    "width": 1440,
                    "height": 1000,
                },
            )

            page = context.new_page()

            # ------------------------------------------------
            # Open page
            # ------------------------------------------------

            print(
                "Opening page..."
            )

            try:

                response = page.goto(
                    TRAINING_URL,
                    wait_until="domcontentloaded",
                    timeout=PAGE_TIMEOUT,
                )

            except PlaywrightTimeoutError:

                print(
                    "Page load timed out."
                )

                send_telegram(
                    build_telegram_message(
                        status="UNKNOWN",
                        confidence=1.0,
                        reason="Page load timed out.",
                        evidence="",
                        url=page.url,
                    )
                )

                return

            except Exception as e:

                print(
                    f"Navigation failed: {e}"
                )

                send_telegram(
                    build_telegram_message(
                        status="UNKNOWN",
                        confidence=1.0,
                        reason=f"Navigation failed: {e}",
                        evidence="",
                        url=page.url,
                    )
                )

                return

            # ------------------------------------------------
            # HTTP status
            # ------------------------------------------------

            if response:

                print(
                    "HTTP status:",
                    response.status,
                )

                if response.status >= 400:

                    send_telegram(
                        build_telegram_message(
                            status="UNKNOWN",
                            confidence=1.0,
                            reason=(
                                f"Website returned HTTP "
                                f"{response.status}."
                            ),
                            evidence="",
                            url=page.url,
                        )
                    )

                    return

            # ------------------------------------------------
            # Allow JavaScript to render
            # ------------------------------------------------

            time.sleep(3)

            current_url = page.url

            print(
                "Final URL:",
                current_url,
            )

            # ------------------------------------------------
            # Extract page text
            # ------------------------------------------------

            page_text = extract_page_text(
                page
            )

            if not page_text:

                send_telegram(
                    build_telegram_message(
                        status="UNKNOWN",
                        confidence=1.0,
                        reason="Page contains no readable text.",
                        evidence="",
                        url=current_url,
                    )
                )

                return

            print(
                "Page text length:",
                len(page_text),
            )

            # ------------------------------------------------
            # Authentication check
            # ------------------------------------------------

            if is_authentication_page(
                current_url,
                page_text,
            ):

                print(
                    "Authentication appears to have expired."
                )

                send_telegram(
                    build_telegram_message(
                        status="UNKNOWN",
                        confidence=1.0,
                        reason=(
                            "The saved authentication state "
                            "appears to have expired."
                        ),
                        evidence=(
                            "Authentication/login page detected."
                        ),
                        url=current_url,
                    )
                )

                return

            # ------------------------------------------------
            # AI classification
            # ------------------------------------------------

            print()
            print(
                "Classifying page..."
            )

            try:

                result = classify_with_ai(
                    page_text
                )

            except Exception as e:

                print(
                    f"AI classification failed: {e}"
                )

                send_telegram(
                    build_telegram_message(
                        status="UNKNOWN",
                        confidence=1.0,
                        reason=(
                            f"AI classification failed: {e}"
                        ),
                        evidence="",
                        url=current_url,
                    )
                )

                return

            status = result["status"]
            confidence = result["confidence"]
            reason = result["reason"]
            evidence = result["evidence"]

            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print()
            print("=" * 70)
            print("RESULT")
            print("=" * 70)
            print()

            print(
                f"Status: {status}"
            )

            print(
                f"Confidence: {confidence:.0%}"
            )

            print(
                f"Reason: {reason}"
            )

            print(
                f"Evidence: {evidence}"
            )

            print()

            # ------------------------------------------------
            # Send current state to Telegram
            # ------------------------------------------------

            message = build_telegram_message(
                status=status,
                confidence=confidence,
                reason=reason,
                evidence=evidence,
                url=current_url,
            )

            send_telegram(
                message
            )

        finally:

            if browser:

                browser.close()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    monitor()