"""Unit tests for Ray Serve SageMaker adapter."""

import importlib
import os
import sys
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "docker", "ray")
)

from sagemaker_serve import app  # noqa: E402


@pytest.fixture
def mock_backend():
    """Provide a helper that patches app.state.client with a mock httpx response."""

    class MockBackend:
        def __init__(self):
            self.last_request_headers = None

        def configure(self, status_code=200, body=b'{"ok":true}', headers=None):
            resp_headers = {"content-type": "application/json"}
            if headers:
                resp_headers.update(headers)

            response = httpx.Response(
                status_code=status_code,
                content=body,
                headers=resp_headers,
            )

            async def mock_post(url, content=None, headers=None):
                self.last_request_headers = headers
                return response

            return mock_post

    return MockBackend()


class TestRequestHeaderForwarding:
    """X-Amzn-SageMaker-* headers flow from SageMaker to Ray Serve."""

    def test_custom_attributes_forwarded(self, mock_backend):
        from starlette.testclient import TestClient

        mock_post = mock_backend.configure()

        with TestClient(app) as client:
            client.app.state.client = AsyncMock()
            client.app.state.client.post = mock_post

            client.post(
                "/invocations",
                content=b"hello",
                headers={
                    "Content-Type": "application/octet-stream",
                    "Accept": "application/json",
                    "X-Amzn-SageMaker-Custom-Attributes": "trace-id-123",
                },
            )

        assert (
            mock_backend.last_request_headers["x-amzn-sagemaker-custom-attributes"]
            == "trace-id-123"
        )

    def test_multiple_sagemaker_headers_forwarded(self, mock_backend):
        from starlette.testclient import TestClient

        mock_post = mock_backend.configure()

        with TestClient(app) as client:
            client.app.state.client = AsyncMock()
            client.app.state.client.post = mock_post

            client.post(
                "/invocations",
                content=b"hello",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Amzn-SageMaker-Custom-Attributes": "my-attr",
                    "X-Amzn-SageMaker-Inference-Id": "inf-001",
                    "X-Amzn-SageMaker-Session-Id": "sess-abc",
                },
            )

        headers = mock_backend.last_request_headers
        assert headers["x-amzn-sagemaker-custom-attributes"] == "my-attr"
        assert headers["x-amzn-sagemaker-inference-id"] == "inf-001"
        assert headers["x-amzn-sagemaker-session-id"] == "sess-abc"

    def test_non_sagemaker_headers_not_forwarded(self, mock_backend):
        from starlette.testclient import TestClient

        mock_post = mock_backend.configure()

        with TestClient(app) as client:
            client.app.state.client = AsyncMock()
            client.app.state.client.post = mock_post

            client.post(
                "/invocations",
                content=b"hello",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Custom-Header": "should-not-forward",
                },
            )

        headers = mock_backend.last_request_headers
        assert "X-Custom-Header" not in headers
        assert "x-custom-header" not in headers

    def test_content_type_and_accept_always_forwarded(self, mock_backend):
        from starlette.testclient import TestClient

        mock_post = mock_backend.configure()

        with TestClient(app) as client:
            client.app.state.client = AsyncMock()
            client.app.state.client.post = mock_post

            client.post(
                "/invocations",
                content=b"data",
                headers={
                    "Content-Type": "image/png",
                    "Accept": "image/jpeg",
                },
            )

        headers = mock_backend.last_request_headers
        assert headers["Content-Type"] == "image/png"
        assert headers["Accept"] == "image/jpeg"


class TestResponseHeaderForwarding:
    """X-Amzn-SageMaker-* headers flow from Ray Serve back to the caller."""

    def test_response_custom_attributes_forwarded(self, mock_backend):
        from starlette.testclient import TestClient

        mock_post = mock_backend.configure(
            headers={"X-Amzn-SageMaker-Custom-Attributes": "resp-attr-456"}
        )

        with TestClient(app) as client:
            client.app.state.client = AsyncMock()
            client.app.state.client.post = mock_post

            response = client.post(
                "/invocations",
                content=b"hello",
                headers={"Content-Type": "application/json"},
            )

        assert response.headers["x-amzn-sagemaker-custom-attributes"] == "resp-attr-456"

    def test_response_multiple_sagemaker_headers_forwarded(self, mock_backend):
        from starlette.testclient import TestClient

        mock_post = mock_backend.configure(
            headers={
                "X-Amzn-SageMaker-Custom-Attributes": "val1",
                "X-Amzn-SageMaker-New-Session-Id": "new-sess",
            }
        )

        with TestClient(app) as client:
            client.app.state.client = AsyncMock()
            client.app.state.client.post = mock_post

            response = client.post(
                "/invocations",
                content=b"hello",
                headers={"Content-Type": "application/json"},
            )

        assert response.headers["x-amzn-sagemaker-custom-attributes"] == "val1"
        assert response.headers["x-amzn-sagemaker-new-session-id"] == "new-sess"

    def test_response_non_sagemaker_headers_not_forwarded(self, mock_backend):
        from starlette.testclient import TestClient

        mock_post = mock_backend.configure(headers={"X-Internal-Debug": "should-not-leak"})

        with TestClient(app) as client:
            client.app.state.client = AsyncMock()
            client.app.state.client.post = mock_post

            response = client.post(
                "/invocations",
                content=b"hello",
                headers={"Content-Type": "application/json"},
            )

        assert "x-internal-debug" not in response.headers


class TestBackendTimeoutConfig:
    """RAYSERVE_BACKEND_TIMEOUT env var controls the httpx proxy timeout."""

    def _reload(self):
        import sagemaker_serve

        return importlib.reload(sagemaker_serve)

    def test_default_is_300(self, monkeypatch):
        monkeypatch.delenv("RAYSERVE_BACKEND_TIMEOUT", raising=False)
        mod = self._reload()
        assert mod.RAYSERVE_BACKEND_TIMEOUT == 300

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_TIMEOUT", "1800")
        mod = self._reload()
        assert mod.RAYSERVE_BACKEND_TIMEOUT == 1800

    def test_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_TIMEOUT", "0")
        with pytest.raises(ValueError, match="positive"):
            self._reload()

    def test_negative_rejected(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_TIMEOUT", "-5")
        with pytest.raises(ValueError, match="positive"):
            self._reload()

    def test_non_integer_rejected(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_TIMEOUT", "abc")
        with pytest.raises(ValueError):
            self._reload()

    def test_fractional_rejected(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_TIMEOUT", "300.5")
        with pytest.raises(ValueError):
            self._reload()


class TestBackendUrlValidation:
    """RAYSERVE_BACKEND_URL must be a well-formed http(s) URL."""

    def _reload(self):
        import sagemaker_serve

        return importlib.reload(sagemaker_serve)

    def test_default_url(self, monkeypatch):
        monkeypatch.delenv("RAYSERVE_BACKEND_URL", raising=False)
        mod = self._reload()
        assert mod.RAYSERVE_URL == "http://127.0.0.1:8000"

    def test_custom_url_accepted(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_URL", "https://ray.internal:9000")
        mod = self._reload()
        assert mod.RAYSERVE_URL == "https://ray.internal:9000"

    def test_missing_scheme_rejected(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_URL", "127.0.0.1:8000")
        with pytest.raises(ValueError, match="invalid"):
            self._reload()

    def test_non_http_scheme_rejected(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_URL", "ftp://ray:8000")
        with pytest.raises(ValueError, match="invalid"):
            self._reload()

    def test_empty_string_rejected(self, monkeypatch):
        monkeypatch.setenv("RAYSERVE_BACKEND_URL", "")
        with pytest.raises(ValueError, match="invalid"):
            self._reload()
