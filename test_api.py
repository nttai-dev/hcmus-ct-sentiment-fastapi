import json
import os
import sys
import requests


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_api_tests(base_url: str = "http://127.0.0.1:8000") -> None:
    base_url = base_url.rstrip("/")

    # 1. Test GET /
    print_section("1. Testing GET / (System Introduction)")
    resp_root = requests.get(f"{base_url}/", timeout=10)
    print(f"Status Code: {resp_root.status_code}")
    print(json.dumps(resp_root.json(), indent=2))
    assert resp_root.status_code == 200, f"Expected 200, got {resp_root.status_code}"
    assert resp_root.json().get("status") == "ok"

    # 2. Test GET /health
    print_section("2. Testing GET /health (Operational Health Check)")
    resp_health = requests.get(f"{base_url}/health", timeout=10)
    print(f"Status Code: {resp_health.status_code}")
    print(json.dumps(resp_health.json(), indent=2))
    assert resp_health.status_code == 200, f"Expected 200, got {resp_health.status_code}"
    assert resp_health.json().get("status") == "healthy"
    assert resp_health.json().get("model_loaded") is True

    # 3. Test POST /predict (Input 1: Positive sentiment)
    print_section("3. Testing POST /predict (Input 1 - Positive Sentiment)")
    payload_pos = {"text": "The lecture on FastAPI was clear, practical, and easy to follow!"}
    resp_pos = requests.post(f"{base_url}/predict", json=payload_pos, timeout=15)
    print(f"Input: {payload_pos['text']}")
    print(f"Status Code: {resp_pos.status_code}")
    print(json.dumps(resp_pos.json(), indent=2))
    assert resp_pos.status_code == 200
    assert resp_pos.json()["prediction"]["label"] == "POSITIVE"

    # 4. Test POST /predict (Input 2: Negative sentiment)
    print_section("4. Testing POST /predict (Input 2 - Negative Sentiment)")
    payload_neg = {"text": "The app kept crashing every five minutes and lost all my saved work."}
    resp_neg = requests.post(f"{base_url}/predict", json=payload_neg, timeout=15)
    print(f"Input: {payload_neg['text']}")
    print(f"Status Code: {resp_neg.status_code}")
    print(json.dumps(resp_neg.json(), indent=2))
    assert resp_neg.status_code == 200
    assert resp_neg.json()["prediction"]["label"] == "NEGATIVE"

    # 5. Test POST /predict (Input 3: Product review sentence)
    print_section("5. Testing POST /predict (Input 3 - Product Review)")
    payload_rev = {"text": "Battery life lasts two full days and the keyboard feels great to type on."}
    resp_rev = requests.post(f"{base_url}/predict", json=payload_rev, timeout=15)
    print(f"Input: {payload_rev['text']}")
    print(f"Status Code: {resp_rev.status_code}")
    print(json.dumps(resp_rev.json(), indent=2))
    assert resp_rev.status_code == 200
    assert resp_rev.json()["prediction"]["label"] == "POSITIVE"

    # 6. Test POST /predict (Validation error: whitespace-only input)
    print_section("6. Testing POST /predict (Validation Error - Empty/Whitespace Input)")
    payload_invalid = {"text": "   "}
    resp_invalid = requests.post(f"{base_url}/predict", json=payload_invalid, timeout=10)
    print(f"Input: {repr(payload_invalid['text'])}")
    print(f"Status Code: {resp_invalid.status_code}")
    print(json.dumps(resp_invalid.json(), indent=2))
    assert resp_invalid.status_code == 422, f"Expected 422, got {resp_invalid.status_code}"

    print("\n" + "=" * 70)
    print("  ALL API TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
    try:
        run_api_tests(target_url)
    except Exception as err:
        print(f"\n[ERROR] API test failed: {err}", file=sys.stderr)
        sys.exit(1)
