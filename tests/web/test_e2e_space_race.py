import pytest
import threading
import time
import socket
import uvicorn
from playwright.sync_api import sync_playwright
from web.server.main import app

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

@pytest.fixture(scope="module")
def server_url():
    port = get_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    
    url = f"http://127.0.0.1:{port}"
    time.sleep(1.0)
    yield url
    server.should_exit = True

@pytest.fixture(scope="module")
def browser_context():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        yield browser
        browser.close()

def test_workbench_page_load_and_header_tracks(browser_context, server_url):
    page = browser_context.new_page(viewport={"width": 1440, "height": 900})
    page.goto(server_url)
    page.wait_for_selector("#space-race-widget", timeout=5000)
    
    # Check title and top-bar tracks
    assert "Twilight Struggle" in page.title()
    assert page.locator(".top-bar").is_visible()
    assert page.locator("#ts-map-svg").is_visible()
    assert page.locator("#decision-panel").is_visible()
    assert page.locator(".defcon-scale").is_visible()
    assert page.locator(".vp-meter-container").is_visible()
    assert page.locator("#space-race-widget").is_visible()
    page.close()

def test_space_race_header_widget_initial_state(browser_context, server_url):
    page = browser_context.new_page(viewport={"width": 1440, "height": 900})
    page.goto(server_url)
    page.wait_for_selector("#space-race-widget", timeout=5000)
    
    widget = page.locator("#space-race-widget")
    assert widget.is_visible()
    
    # Verify USSR pill
    ussr_pill = widget.locator(".space-pill.ussr")
    assert ussr_pill.is_visible()
    assert "#0" in ussr_pill.locator(".pill-box-num").inner_text()
    assert "(0/1)" in ussr_pill.locator(".pill-attempts").inner_text()
    
    # Verify US pill
    us_pill = widget.locator(".space-pill.us")
    assert us_pill.is_visible()
    assert "#0" in us_pill.locator(".pill-box-num").inner_text()
    assert "(0/1)" in us_pill.locator(".pill-attempts").inner_text()
    
    # Verify mini progress ladder
    mini_boxes = widget.locator(".mini-box")
    assert mini_boxes.count() == 9
    
    # Box 0 should be active for both players initially
    box0 = widget.locator('.mini-box[data-step="0"]')
    assert "active" in box0.get_attribute("class")
    assert "both" in box0.get_attribute("class")
    page.close()

def test_space_race_modal_interaction_and_content(browser_context, server_url):
    page = browser_context.new_page(viewport={"width": 1440, "height": 900})
    page.goto(server_url)
    page.wait_for_selector("#space-race-widget", timeout=5000)
    
    # Click header widget to open modal
    page.locator("#space-race-widget").click()
    page.wait_for_selector("#modal-container:not(.hidden)", timeout=3000)
    
    modal_title = page.locator("#modal-title").inner_text()
    assert "DELUXE SPACE RACE TRACK & PERKS" in modal_title
    
    # Verify all 9 space cards
    cards = page.locator(".space-box-card")
    assert cards.count() == 9
    
    # Verify Box 1 (Earth Satellite)
    box1 = page.locator('.space-box-card[data-box="1"]')
    assert "Earth Satellite" in box1.inner_text()
    assert "≥ 2 Ops" in box1.inner_text()
    assert "1–3" in box1.inner_text()
    assert "+2 VP" in box1.inner_text()
    
    # Verify Box 2 (Animal in Space)
    box2 = page.locator('.space-box-card[data-box="2"]')
    assert "Animal in Space" in box2.inner_text()
    assert "May attempt Space Race twice per turn" in box2.inner_text()
    
    # Verify Box 4 (Man in Space)
    box4 = page.locator('.space-box-card[data-box="4"]')
    assert "Man in Space" in box4.inner_text()
    assert "Opponent must select & reveal headline first" in box4.inner_text()
    
    # Verify Box 6 (Space Walk)
    box6 = page.locator('.space-box-card[data-box="6"]')
    assert "Space Walk" in box6.inner_text()
    assert "≥ 3 Ops" in box6.inner_text()
    assert "May discard 1 held card at end of turn" in box6.inner_text()
    
    # Verify Box 8 (Eagle / Bear Landed)
    box8 = page.locator('.space-box-card[data-box="8"]')
    assert "Eagle / Bear Landed" in box8.inner_text()
    assert "≥ 4 Ops" in box8.inner_text()
    assert "1–2" in box8.inner_text()
    assert "May play 8 Action Rounds per turn" in box8.inner_text()
    
    # Test close button
    page.locator("#modal-close").click()
    page.wait_for_selector("#modal-container", state="hidden", timeout=3000)
    
    # Test Escape key closes modal
    page.locator("#space-race-widget").click()
    page.wait_for_selector("#modal-container:not(.hidden)", timeout=3000)
    page.keyboard.press("Escape")
    page.wait_for_selector("#modal-container", state="hidden", timeout=3000)
    page.close()

def test_space_race_replay_state_synchronization(browser_context, server_url):
    page = browser_context.new_page(viewport={"width": 1440, "height": 900})
    page.goto(server_url)
    page.wait_for_selector("#rep-server-select", timeout=5000)
    
    # Wait for replay dropdown options to populate
    page.wait_for_function('document.getElementById("rep-server-select").options.length > 1', timeout=5000)
    server_select = page.locator("#rep-server-select")
    server_select.select_option(value="strategic_llm_game_defcon2.tslog.json")
    
    # Wait for replay slider to be ready
    page.wait_for_function('parseInt(document.getElementById("rep-timeline-slider").max, 10) > 100', timeout=5000)
    
    # Jump to Step 155 (US Box 2, USSR Box 1)
    page.evaluate('window.__wb.replayControls.goToStep(155)')
    page.wait_for_timeout(300)
    
    # Verify Space Race header widget shows US: #2 and USSR: #1
    widget = page.locator("#space-race-widget")
    assert "#1" in widget.locator(".space-pill.ussr .pill-box-num").inner_text()
    assert "#2" in widget.locator(".space-pill.us .pill-box-num").inner_text()
    
    # Verify mini ladder reflects US at 2 and USSR at 1
    box1 = widget.locator('.mini-box[data-step="1"]')
    box2 = widget.locator('.mini-box[data-step="2"]')
    assert "ussr" in box1.get_attribute("class")
    assert "us" in box2.get_attribute("class")
    
    # Open modal and verify active perks display
    widget.click()
    page.wait_for_selector("#modal-container:not(.hidden)", timeout=3000)
    
    lead_badge = page.locator(".lead-badge")
    assert "US LEADS BY 1 SPACE" in lead_badge.inner_text()
    
    # Check Box 2 active privilege for US (Animal in Space)
    box2_card = page.locator('.space-box-card[data-box="2"]')
    assert "ACTIVE FOR US" in box2_card.inner_text()
    
    page.close()
