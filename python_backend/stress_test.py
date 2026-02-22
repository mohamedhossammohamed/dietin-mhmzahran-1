import asyncio
import httpx
import time
import subprocess
import sys

async def fetch(client, url):
    try:
        response = await client.get(url, timeout=5.0)
        return response.status_code
    except Exception as e:
        return str(e)

async def main():
    print("Starting server...")
    server_process = subprocess.Popen(
        ["venv/bin/uvicorn", "main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(10)
    
    if server_process.poll() is not None:
        print("Server crashed with return code:", server_process.returncode)
        print("STDERR:", server_process.stderr.read().decode())
        return

    url = "http://localhost:8000/health"
    print("Starting stress test...")
    start_time = time.time()
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000)) as client:
        tasks = [fetch(client, url) for _ in range(1000)]
        results = await asyncio.gather(*tasks)
        
    end_time = time.time()
    success_count = sum(1 for r in results if r == 200)
    error_count = 1000 - success_count
    
    print(f"Time: {end_time - start_time:.2f}s, Success: {success_count}, Fail: {error_count}")
    server_process.terminate()

if __name__ == "__main__":
    asyncio.run(main())
