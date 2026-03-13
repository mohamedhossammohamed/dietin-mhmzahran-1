import pytest
import tempfile
import asyncio
from fastapi import UploadFile, HTTPException
from fastapi.datastructures import Headers
from main import validate_file_size

@pytest.mark.asyncio
async def test_validate_file_size_valid():
    """Test that a file within the allowed size limit does not raise an exception."""
    max_size_mb = 5
    # Create a 1MB file
    file_content = b"0" * (1 * 1024 * 1024)

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(file_content)
        tmp.flush()

        with open(tmp.name, 'rb') as f:
            upload_file = UploadFile(
                filename="test_valid.txt",
                file=f,
                size=len(file_content),
                headers=Headers({"content-type": "text/plain"})
            )

            # This should not raise any exception
            await validate_file_size(upload_file, max_size_mb)

@pytest.mark.asyncio
async def test_validate_file_size_exceeds_limit():
    """Test that a file exceeding the size limit raises an HTTPException with status code 413."""
    max_size_mb = 1
    # Create a 2MB file
    file_content = b"0" * (2 * 1024 * 1024)

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(file_content)
        tmp.flush()

        with open(tmp.name, 'rb') as f:
            upload_file = UploadFile(
                filename="test_invalid.txt",
                file=f,
                size=len(file_content),
                headers=Headers({"content-type": "text/plain"})
            )

            # Check if HTTPException is raised with 413
            with pytest.raises(HTTPException) as exc_info:
                await validate_file_size(upload_file, max_size_mb)

            assert exc_info.value.status_code == 413
            assert exc_info.value.detail == "Payload Too Large"
