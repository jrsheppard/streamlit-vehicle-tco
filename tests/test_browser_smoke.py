"""Browser-level smoke test for rendered behavior only.

AppTest cannot see layout, so this checks what only a real browser can: the page
renders without a Streamlit exception, charts and tables actually draw, and the
page does not overflow horizontally on a desktop or a narrow viewport.

It is skipped unless Playwright and its Chromium browser are installed::

    venv/bin/python -m pip install -r requirements-dev.txt
    venv/bin/python -m playwright install chromium
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESKTOP = {"width": 1440, "height": 900}
NARROW = {"width": 390, "height": 844}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def app_server() -> str:
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if process.poll() is not None:
            pytest.fail("The Streamlit server exited before it became reachable.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail("The Streamlit server did not start in time.")
    try:
        yield url
    finally:
        process.terminate()
        process.wait(timeout=30)


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch()
        except Exception as error:  # browser binary not installed
            pytest.skip(f"Chromium is not available: {error}")
        try:
            yield instance
        finally:
            instance.close()


def open_app(browser, url: str, viewport: dict[str, int]):
    page = browser.new_page(viewport=viewport)
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("text=Vehicle total cost of ownership", timeout=60_000)
    page.wait_for_selector("[data-testid='stDataFrame']", timeout=60_000)
    return page


@pytest.mark.parametrize("viewport", [DESKTOP, NARROW], ids=["desktop", "narrow"])
def test_page_renders_without_overflow_or_exceptions(app_server, browser, viewport):
    page = open_app(browser, app_server, viewport)
    try:
        assert page.locator("[data-testid='stException']").count() == 0

        page.wait_for_selector(
            "[data-testid='stVegaLiteChart'] :is(svg, canvas)", timeout=60_000
        )
        assert page.locator("[data-testid='stVegaLiteChart']").count() >= 2
        assert page.locator("[data-testid='stMetric']").count() >= 5
        marks = page.evaluate(
            "() => document.querySelectorAll(\"[data-testid='stVegaLiteChart'] svg path,"
            " [data-testid='stVegaLiteChart'] svg rect\").length"
        )
        assert marks > 0, "The charts rendered no marks."

        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - window.innerWidth"
        )
        assert overflow <= 2, f"The page overflows horizontally by {overflow}px."

        chart_box = page.locator("[data-testid='stVegaLiteChart']").first.bounding_box()
        assert chart_box is not None
        assert chart_box["width"] > 100
        assert chart_box["height"] > 100
        assert chart_box["x"] + chart_box["width"] <= viewport["width"] + 2

        table_box = page.locator("[data-testid='stDataFrame']").first.bounding_box()
        assert table_box is not None
        assert table_box["height"] > 80
    finally:
        page.close()
