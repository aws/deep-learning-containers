"""Unit tests for SageMaker routing middleware."""

import asyncio
import os
import sys

# Allow importing omni_sagemaker_serve from scripts/vllm/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "vllm"))

import pytest
from omni_sagemaker_serve import SageMakerRouteMiddleware, _parse_route


class TestParseRoute:
    def test_extracts_route(self):
        headers = [(b"x-amzn-sagemaker-custom-attributes", b"route=/v1/audio/speech")]
        assert _parse_route(headers) == "/v1/audio/speech"

    def test_extracts_route_with_extra_attrs(self):
        headers = [(b"x-amzn-sagemaker-custom-attributes", b"foo=bar,route=/v1/audio/speech,baz=1")]
        assert _parse_route(headers) == "/v1/audio/speech"

    def test_no_route(self):
        headers = [(b"x-amzn-sagemaker-custom-attributes", b"foo=bar")]
        assert _parse_route(headers) is None

    def test_no_header(self):
        assert _parse_route([]) is None

    def test_case_insensitive_header(self):
        headers = [(b"X-Amzn-SageMaker-Custom-Attributes", b"route=/v1/chat/completions")]
        assert _parse_route(headers) == "/v1/chat/completions"


class TestMiddleware:
    @pytest.fixture
    def captured(self):
        return {}

    @pytest.fixture
    def app(self, captured):
        async def inner(scope, receive, send):
            captured["path"] = scope["path"]

        return inner

    @pytest.fixture
    def middleware(self, app):
        return SageMakerRouteMiddleware(app)

    def _make_scope(self, path="/invocations", headers=None):
        return {
            "type": "http",
            "path": path,
            "raw_path": path.encode(),
            "headers": headers or [],
        }

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_rewrites_with_route_header(self, middleware, captured):
        scope = self._make_scope(
            headers=[
                (b"x-amzn-sagemaker-custom-attributes", b"route=/v1/audio/speech"),
            ]
        )
        self._run(middleware(scope, None, None))
        assert captured["path"] == "/v1/audio/speech"

    def test_falls_through_without_route(self, middleware, captured):
        scope = self._make_scope()
        self._run(middleware(scope, None, None))
        assert captured["path"] == "/invocations"

    def test_ignores_non_invocations(self, middleware, captured):
        scope = self._make_scope(path="/health")
        self._run(middleware(scope, None, None))
        assert captured["path"] == "/health"

    def test_ignores_non_http(self, middleware, captured):
        scope = {"type": "websocket", "path": "/invocations"}
        self._run(middleware(scope, None, None))
        assert captured["path"] == "/invocations"

    def test_rewrites_raw_path(self, middleware, captured):
        scope = self._make_scope(
            headers=[
                (b"x-amzn-sagemaker-custom-attributes", b"route=/v1/completions"),
            ]
        )
        self._run(middleware(scope, None, None))
        assert captured["path"] == "/v1/completions"

    def test_adapter_attrs_without_route_falls_through(self, middleware, captured):
        """Adapter attributes (no route=) should fall through to /invocations."""
        scope = self._make_scope(
            headers=[
                (b"x-amzn-sagemaker-custom-attributes", b"adapter=my-lora-adapter"),
            ]
        )
        self._run(middleware(scope, None, None))
        assert captured["path"] == "/invocations"

    def test_adapter_attrs_with_route_rewrites(self, middleware, captured):
        """Both adapter and route attrs — route takes effect, adapter preserved in headers."""
        scope = self._make_scope(
            headers=[
                (
                    b"x-amzn-sagemaker-custom-attributes",
                    b"adapter=my-lora,route=/v1/audio/speech",
                ),
            ]
        )
        self._run(middleware(scope, None, None))
        assert captured["path"] == "/v1/audio/speech"
