"""
Phase 1 Stress Tests — Automated acceptance criteria verification.
Run: python -m pytest test_phase1.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Import only the FastAPI app — Phase 2-5 modules may or may not be present
from main import app

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Health Endpoint — Schema & Values
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Master Plan Phase 1, item 6: GET /health returns status + device."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_keys(self):
        data = client.get("/health").json()
        assert "status" in data, "Missing 'status' key"
        assert "device" in data, "Missing 'device' key"

    def test_health_status_is_healthy(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_device_is_valid(self):
        data = client.get("/health").json()
        assert data["device"] in ("mps", "cuda", "cpu"), (
            f"Unexpected device: {data['device']}"
        )

    def test_health_no_extra_keys(self):
        data = client.get("/health").json()
        assert set(data.keys()) == {"status", "device"}, (
            f"Unexpected keys: {set(data.keys()) - {'status', 'device'}}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. CORS — Only allowed origins pass
# ═══════════════════════════════════════════════════════════════════════════


class TestCORS:
    """Master Plan Phase 1, item 5: CORS for localhost:3000 and :5173 only."""

    @pytest.mark.parametrize(
        "origin",
        ["http://localhost:3000", "http://localhost:5173"],
    )
    def test_allowed_origins(self, origin):
        response = client.options(
            "/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == origin

    @pytest.mark.parametrize(
        "origin",
        [
            "http://evil.com",
            "http://localhost:9999",
            "https://localhost:3000",  # wrong scheme
        ],
    )
    def test_disallowed_origins(self, origin):
        response = client.options(
            "/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        acao = response.headers.get("access-control-allow-origin")
        # Must NOT echo back a disallowed origin or use wildcard
        assert acao != origin and acao != "*", (
            f"Origin {origin} should be blocked but got ACAO={acao}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Routing — 404 for undefined routes
# ═══════════════════════════════════════════════════════════════════════════


class TestRouting:
    def test_undefined_route_returns_404(self):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_health_wrong_method(self):
        response = client.post("/health")
        assert response.status_code == 405


# ═══════════════════════════════════════════════════════════════════════════
# 4. Concurrency — Multiple rapid requests
# ═══════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    def test_rapid_sequential_requests(self):
        """10 rapid requests should all return 200 + healthy."""
        for _ in range(10):
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "healthy"
