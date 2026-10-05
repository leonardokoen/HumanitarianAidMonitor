from pathlib import Path
from playwright.sync_api import sync_playwright

TRAINING_URL = "https://youth.europa.eu/solidarity/dashboard/training-humanitarian-aid_en"

AUTH_FILE = Path("auth_state.json")

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    context = browser.new_context(
        storage_state=str(AUTH_FILE.absolute())
    )

    page = context.new_page()

    page.goto(
        TRAINING_URL,
        wait_until="domcontentloaded",
        timeout=60_000,
    )

    print()
    print("URL:", page.url)
    print()

    print("PAGE TEXT:")
    print("=" * 80)
    print(page.locator("body").inner_text())
    print("=" * 80)

    input("\nPress ENTER to close...")

    browser.close()