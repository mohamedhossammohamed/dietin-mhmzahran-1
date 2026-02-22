import requests
import json
import time
import os

BASE_URL = "http://localhost:8000/api/v1"

def print_result(test_name, expected, res):
    print(f"\n{'='*50}")
    print(f"TEST: {test_name}")
    print(f"Status Code: {res.status_code}")
    try:
        data = res.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        
        # Check warnings for Liquid test
        if "Liquid" in test_name and data.get("data", {}).get("warnings"):
            print("WARNINGS:", data["data"]["warnings"])
            
    except:
        print("Response Text:", res.text)
    print(f"{'='*50}\n")

# Test 1: Happy Path (Solid Food)
def test_1():
    files = {"full_image": open("test_apple.jpg", "rb")}
    res = requests.post(f"{BASE_URL}/analyze/image", files=files)
    print_result("TEST 1: Happy Path (Solid Food)", "Should return valid macros and high confidence", res)

# Test 2: Liquid Fork (Soup Problem)
def test_2():
    files = {"full_image": open("test_coffee.jpg", "rb")}
    res = requests.post(f"{BASE_URL}/analyze/image", files=files)
    print_result("TEST 2: Liquid Fork (Clear Liquid)", "Should bypass metric depth, return 0 calories, confidence 0, and show warning", res)

# Test 3: Nutrition Label
def test_3():
    files = {"file": open("test_label.jpg", "rb")}
    res = requests.post(f"{BASE_URL}/analyze/label", files=files)
    print_result("TEST 3: Nutrition Label", "Should use OCR, bypass physics engine, return exactly label macros", res)

# Test 4: Maximum Payload Error Shield
def test_4():
    files = {"full_image": open("test_massive.jpg", "rb")}
    res = requests.post(f"{BASE_URL}/analyze/image", files=files)
    print_result("TEST 4: VRAM/OOM Protection", "Should fail cleanly with 413 Payload Too Large", res)

# Test 5: Audio Transcription (Phase 5)
def test_5():
    files = {"audio_file": open("test_audio.m4a", "rb")}
    res = requests.post(f"{BASE_URL}/analyze/speech", files=files)
    print_result("TEST 5: Audio Transcription", "Should transcribe 'fried chicken', query ChromaDB, return macros", res)

if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
    if os.path.exists("test_audio.m4a"):
        test_5()
    else:
        print("Skip Test 5: No audio file found")
