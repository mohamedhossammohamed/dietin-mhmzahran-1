import requests

URL = "http://localhost:8000/api/v1/analyze/image"

print("Test 1: Empty Payload")
r = requests.post(URL)
print(r.status_code, r.text)

print("\nTest 2: Fake Image Data")
with open("test.jpg", "wb") as f:
    f.write(b"fake image data this is not an image")
r = requests.post(URL, files={"file": open("test.jpg", "rb")})
print(r.status_code, r.text)

print("\nTest 3: Mode 2 Crops")
r = requests.post(URL, files={
    "context_image": ("context.jpg", b"fake"),
    "crops": ("crop1.jpg", b"fake crop")
})
print(r.status_code, r.text)

