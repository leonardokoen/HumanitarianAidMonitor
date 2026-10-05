from pathlib import Path
from playwright.sync_api import sync_playwright

TRAINING_URL = "https://youth.europa.eu/solidarity/dashboard/training-humanitarian-aid_en"

PROFILE_DIR = Path("playwright_profile")
AUTH_FILE = Path("auth_state.json")

with sync_playwright() as p:

    browser = p.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR.absolute()),
        headless=False,
    )

    page = browser.pages[0] if browser.pages else browser.new_page()

    page.goto(
        TRAINING_URL,
        wait_until="domcontentloaded",
        timeout=60_000,
    )

    print()
    print("==========================================")
    print("Log in to the European Youth Portal")
    print("==========================================")
    print()
    print("When you can see the Humanitarian Aid")
    print("training page, come back here.")
    print()

    input("Press ENTER after you have logged in...")

    # Save cookies + local storage
    context = page.context

    context.storage_state(
        path=str(AUTH_FILE.absolute())
    )

    print()
    print("Authentication state saved!")
    print()
    print(f"File: {AUTH_FILE.absolute()}")
    print()

    browser.close()