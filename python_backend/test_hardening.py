import asyncio
import os
from fastapi import UploadFile, HTTPException
from fastapi.datastructures import Headers
import tempfile
import sys

sys.path.append('/Users/mohammedhossam/Desktop/dietin-mhmzahran-1/python_backend')
from main import validate_file_size

async def test_payload_cap():
    # Test valid size (e.g. 5MB limit)
    with tempfile.NamedTemporaryFile() as tmp:
        # write 1MB of data
        tmp.write(b"0" * 1024 * 1024)
        tmp.flush()
        with open(tmp.name, 'rb') as f:
            upload_file = UploadFile(filename="test.txt", file=f, size=1024*1024, headers=Headers({"content-type": "text/plain"}))
            try:
                await validate_file_size(upload_file, 5)
                print("Test 1 Passed: Valid size accepted")
            except Exception as e:
                print(f"Test 1 Failed: {e}")
                
    # Test invalid size (e.g. 1MB limit, 2MB file)
    with tempfile.NamedTemporaryFile() as tmp:
        # write 2MB of data
        tmp.write(b"0" * 2 * 1024 * 1024)
        tmp.flush()
        with open(tmp.name, 'rb') as f:
            upload_file = UploadFile(filename="test2.txt", file=f, size=2*1024*1024, headers=Headers({"content-type": "text/plain"}))
            try:
                await validate_file_size(upload_file, 1)
                print("Test 2 Failed: Oversized file accepted")
            except HTTPException as e:
                if e.status_code == 413:
                    print("Test 2 Passed: 413 Payload Too Large raised correctly")
                else:
                    print(f"Test 2 Failed: Wrong exception {e}")
            except Exception as e:
                print(f"Test 2 Failed: Wrong exception type {type(e)}")

if __name__ == "__main__":
    asyncio.run(test_payload_cap())
