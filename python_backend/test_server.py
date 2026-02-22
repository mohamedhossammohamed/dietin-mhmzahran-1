import requests
import traceback

def test_server():
    try:
        r = requests.post("http://localhost:8000/api/v1/analyze/image")
        print("Empty payload status:", r.status_code)
        print("Empty payload text:", r.text)

        with open("dummy.jpg", "wb") as f:
            f.write(b"dummy")
        
        r2 = requests.post("http://localhost:8000/api/v1/analyze/image", files={"file": ("dummy.jpg", open("dummy.jpg", "rb"), "image/jpeg")})
        print("Dummy file status:", r2.status_code)
        print("Dummy file text:", r2.text)
    except Exception as e:
        traceback.print_exc()

test_server()
