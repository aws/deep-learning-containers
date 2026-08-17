"""Unit tests for SageMaker routing middleware."""

import asyncio
import os
import re
import sys

# Allow importing omni_sagemaker_serve from scripts/docker/vllm/
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "docker", "vllm")
)

import json

import pytest
from omni_sagemaker_serve import (
    FORM_DATA_ROUTES,
    WS_ALLOWED_PATHS,
    WS_DEFAULT_PATH,
    SageMakerRouteMiddleware,
    _build_multipart,
    _parse_route,
)


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

    def test_ignores_non_http_non_ws(self, middleware, captured):
        # lifespan (and any non-http, non-websocket scope) passes straight
        # through to the app untouched. (WebSocket scopes are handled by the
        # dedicated branch — see TestWebSocketBridge.)
        scope = {"type": "lifespan", "path": "/invocations"}
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

    def test_multipart_body_passthrough(self, captured):
        """Multipart bodies pass through unchanged — middleware only rewrites path."""
        body_captured = {}

        async def app(scope, receive, send):
            captured["path"] = scope["path"]
            captured["headers"] = dict(scope["headers"])
            msg = await receive()
            body_captured["body"] = msg["body"]

        middleware = SageMakerRouteMiddleware(app)
        form_body = b'--boundary\r\nContent-Disposition: form-data; name="prompt"\r\n\r\na dog\r\n--boundary--\r\n'

        async def receive():
            return {"type": "http.request", "body": form_body, "more_body": False}

        scope = self._make_scope(
            headers=[
                (b"x-amzn-sagemaker-custom-attributes", b"route=/v1/videos"),
                (b"content-type", b"multipart/form-data; boundary=boundary"),
            ]
        )
        self._run(middleware(scope, receive, None))
        assert captured["path"] == "/v1/videos"
        assert body_captured["body"] == form_body
        assert captured["headers"][b"content-type"] == b"multipart/form-data; boundary=boundary"


class TestWebSocketBridge:
    """SageMaker Bidirectional Streaming -> vLLM-Omni WebSocket routes.

    SageMaker's Sidecar substitutes the client's model_invocation_path into the
    container WS URI, so scope["path"] arrives already resolved. The middleware
    lets whitelisted routes through to the app and rejects everything else
    (including the un-overridden default) before the handshake completes.
    """

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _drive(self, path):
        """Drive the middleware with a websocket scope at `path`.

        Returns (passed_through, sent) where passed_through is True if the inner
        app ran (connection accepted for routing), and `sent` is the list of
        ASGI events the middleware emitted (a websocket.close on reject).
        """
        state = {"passed_through": False}
        sent = []

        async def app(scope, receive, send):
            state["passed_through"] = True

        # receive yields the connect event, then disconnect (for the reject path
        # which consumes one event).
        events = iter([{"type": "websocket.connect"}, {"type": "websocket.disconnect"}])

        async def receive():
            return next(events)

        async def send(msg):
            sent.append(msg)

        middleware = SageMakerRouteMiddleware(app)
        scope = {"type": "websocket", "path": path, "headers": []}
        self._run(middleware(scope, receive, send))
        return state["passed_through"], sent

    @pytest.mark.parametrize("path", sorted(WS_ALLOWED_PATHS))
    def test_allowed_ws_routes_pass_through(self, path):
        passed, sent = self._drive(path)
        assert passed is True, f"{path} should pass through to the app"
        # Pass-through must NOT emit a close — the downstream handler owns the
        # handshake (accept/close).
        assert sent == []

    def test_default_path_rejected(self):
        passed, sent = self._drive(WS_DEFAULT_PATH)
        assert passed is False, "un-overridden default path must not reach the app"
        assert len(sent) == 1
        assert sent[0]["type"] == "websocket.close"

    def test_unknown_path_rejected(self):
        passed, sent = self._drive("/v1/not/a/real/route")
        assert passed is False
        assert sent[0]["type"] == "websocket.close"

    def test_reject_consumes_connect_before_close(self):
        """A spec-conformant pre-accept reject: consume websocket.connect, then
        send websocket.close WITHOUT a preceding websocket.accept."""
        _, sent = self._drive(WS_DEFAULT_PATH)
        assert all(m["type"] != "websocket.accept" for m in sent)
        assert sent[-1]["type"] == "websocket.close"

    def test_allowed_paths_match_upstream_router_decorators(self):
        """Guard against WS_ALLOWED_PATHS drifting from the actual vLLM-Omni
        @router.websocket routes. These six are the routes registered in
        vllm_omni/entrypoints/openai/api_server.py at v0.26.0."""
        assert WS_ALLOWED_PATHS == {
            "/v1/audio/speech/stream",
            "/v1/realtime",
            "/v1/realtime/video",
            "/v1/video/chat/stream",
            "/v1/duplex",
            "/v1/realtime/robot/openpi",
        }


def _parse_multipart(body: bytes, content_type: str) -> dict:
    """Minimal multipart/form-data parser for test assertions.

    Splits on the boundary from the content-type and extracts
    name -> value for each simple form-data part.
    """
    boundary = content_type.split("boundary=", 1)[1]
    delimiter = f"--{boundary}".encode()
    fields = {}
    for part in body.split(delimiter):
        if not part.strip() or part.strip() == b"--":
            continue
        head, _, value = part.partition(b"\r\n\r\n")
        m = re.search(rb'name="([^"]+)"', head)
        if m:
            fields[m.group(1).decode()] = value.rstrip(b"\r\n").decode()
    return fields


class TestJsonToMultipart:
    """JSON bodies targeting a form-data route are converted to multipart."""

    def _capture(self, route, body, content_type=b"application/json", chunks=None):
        captured = {}

        async def app(scope, receive, send):
            captured["path"] = scope["path"]
            captured["headers"] = dict(scope["headers"])
            collected = b""
            more = True
            while more:
                msg = await receive()
                collected += msg.get("body", b"")
                more = msg.get("more_body", False)
            captured["body"] = collected

        middleware = SageMakerRouteMiddleware(app)

        # When `chunks` is provided, deliver the body across multiple
        # http.request messages to exercise the streamed-body drain loop.
        messages = (
            [
                {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
                for i, c in enumerate(chunks)
            ]
            if chunks is not None
            else [{"type": "http.request", "body": body, "more_body": False}]
        )
        msg_iter = iter(messages)

        async def receive():
            return next(msg_iter)

        scope = {
            "type": "http",
            "path": "/invocations",
            "raw_path": b"/invocations",
            "headers": [
                (b"x-amzn-sagemaker-custom-attributes", f"route={route}".encode()),
                (b"content-type", content_type),
            ],
        }
        asyncio.get_event_loop().run_until_complete(middleware(scope, receive, None))
        return captured

    def test_video_sync_json_converted_to_multipart(self):
        payload = {
            "prompt": "A cat playing with a ball",
            "num_frames": 17,
            "num_inference_steps": 30,
            "size": "480x320",
        }
        captured = self._capture("/v1/videos/sync", json.dumps(payload).encode())

        assert captured["path"] == "/v1/videos/sync"
        ctype = captured["headers"][b"content-type"].decode()
        assert ctype.startswith("multipart/form-data; boundary=")
        fields = _parse_multipart(captured["body"], ctype)
        assert fields == {
            "prompt": "A cat playing with a ball",
            "num_frames": "17",
            "num_inference_steps": "30",
            "size": "480x320",
        }

    def test_content_length_matches_new_body(self):
        captured = self._capture("/v1/videos/sync", json.dumps({"prompt": "hi"}).encode())
        assert captured["headers"][b"content-length"] == str(len(captured["body"])).encode()

    def test_async_videos_route_also_converted(self):
        assert "/v1/videos" in FORM_DATA_ROUTES
        captured = self._capture("/v1/videos", json.dumps({"prompt": "hi"}).encode())
        assert captured["headers"][b"content-type"].decode().startswith("multipart/form-data")

    def test_bool_and_nested_values_rendered(self):
        payload = {"flag": True, "opts": {"seed": 42}}
        captured = self._capture("/v1/videos/sync", json.dumps(payload).encode())
        ctype = captured["headers"][b"content-type"].decode()
        fields = _parse_multipart(captured["body"], ctype)
        assert fields["flag"] == "true"
        assert json.loads(fields["opts"]) == {"seed": 42}

    def test_non_form_route_json_passes_through(self):
        """JSON to a non-form route (e.g. TTS) is left untouched."""
        body = json.dumps({"input": "hello"}).encode()
        captured = self._capture("/v1/audio/speech", body)
        assert captured["path"] == "/v1/audio/speech"
        assert captured["headers"][b"content-type"] == b"application/json"
        assert captured["body"] == body

    def test_multipart_to_form_route_passes_through(self):
        """An explicit multipart body to a form route is not re-encoded."""
        form_body = (
            b'--b\r\nContent-Disposition: form-data; name="prompt"\r\n\r\na dog\r\n--b--\r\n'
        )
        captured = self._capture(
            "/v1/videos/sync",
            form_body,
            content_type=b"multipart/form-data; boundary=b",
        )
        assert captured["body"] == form_body
        assert captured["headers"][b"content-type"] == b"multipart/form-data; boundary=b"

    def test_invalid_json_passes_through_untouched(self):
        """Malformed JSON is replayed verbatim so upstream returns its own error."""
        bad = b"{not valid json"
        captured = self._capture("/v1/videos/sync", bad)
        assert captured["body"] == bad
        # content-type unchanged because conversion bailed out
        assert captured["headers"][b"content-type"] == b"application/json"

    def test_non_object_json_passes_through_untouched(self):
        """A JSON array (not an object) cannot become form fields -> passthrough."""
        arr = b'["a", "b"]'
        captured = self._capture("/v1/videos/sync", arr)
        assert captured["body"] == arr
        assert captured["headers"][b"content-type"] == b"application/json"

    def test_chunked_json_body_is_reassembled(self):
        """A JSON body split across multiple http.request messages is drained
        in full before conversion."""
        payload = {"prompt": "a dog running on a beach", "num_frames": 17}
        raw = json.dumps(payload).encode()
        mid = len(raw) // 2
        captured = self._capture("/v1/videos/sync", raw, chunks=[raw[:mid], raw[mid:]])
        ctype = captured["headers"][b"content-type"].decode()
        assert ctype.startswith("multipart/form-data")
        fields = _parse_multipart(captured["body"], ctype)
        assert fields == {"prompt": "a dog running on a beach", "num_frames": "17"}

    def test_null_value_field_is_dropped(self):
        """JSON null has no form-data equivalent — the key is omitted so the
        upstream handler applies its default-on-missing behavior rather than
        receiving the literal string "null"."""
        payload = {"prompt": "a dog", "seed": None}
        captured = self._capture("/v1/videos/sync", json.dumps(payload).encode())
        ctype = captured["headers"][b"content-type"].decode()
        fields = _parse_multipart(captured["body"], ctype)
        assert fields == {"prompt": "a dog"}
        assert "seed" not in fields

    def test_field_name_with_quote_is_escaped(self):
        """A key containing a double-quote must not corrupt the
        Content-Disposition header."""
        captured = self._capture("/v1/videos/sync", json.dumps({'a"b': "v"}).encode())
        body = captured["body"]
        # The raw quote in the key is backslash-escaped, so there is exactly
        # one well-formed name="..." part and the upstream parser sees a"b.
        assert b'name="a\\"b"' in body


class TestBuildMultipart:
    def test_roundtrip(self):
        boundary = "abc123"
        body = _build_multipart({"prompt": "hi", "n": "1"}, boundary)
        fields = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        assert fields == {"prompt": "hi", "n": "1"}


class TestVideoInvocationContentTypes:
    """Coverage of the two supported content-type combinations a SageMaker
    caller uses against the video routes:

      1. multipart/form-data + route=/v1/videos/sync  -> passthrough
      2. application/json     + route=/v1/videos/sync  -> converted to multipart

    Both require route=/v1/videos/sync. JSON without a route is not supported
    (it falls through to vLLM's chat handler); that fall-through is already
    covered by TestMiddleware::test_falls_through_without_route.

    Asserts what the upstream vllm-omni handler ends up receiving (path,
    content-type, body), since the upstream handler only accepts
    multipart/form-data.
    """

    VIDEO_FIELDS = {
        "prompt": "A cat playing with a ball",
        "num_frames": 17,
        "num_inference_steps": 30,
        "size": "480x320",
    }

    def _capture(self, headers, body):
        captured = {}

        async def app(scope, receive, send):
            captured["path"] = scope["path"]
            captured["headers"] = dict(scope["headers"])
            collected = b""
            more = True
            while more:
                msg = await receive()
                collected += msg.get("body", b"")
                more = msg.get("more_body", False)
            captured["body"] = collected

        middleware = SageMakerRouteMiddleware(app)

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        scope = {
            "type": "http",
            "path": "/invocations",
            "raw_path": b"/invocations",
            "headers": headers,
        }
        asyncio.get_event_loop().run_until_complete(middleware(scope, receive, None))
        return captured

    def test_case1_multipart_with_route_passthrough(self):
        """multipart/form-data + route -> reaches /v1/videos/sync unchanged."""
        boundary = "case1boundary"
        body = _build_multipart({k: str(v) for k, v in self.VIDEO_FIELDS.items()}, boundary)
        captured = self._capture(
            headers=[
                (b"x-amzn-sagemaker-custom-attributes", b"route=/v1/videos/sync"),
                (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            ],
            body=body,
        )
        assert captured["path"] == "/v1/videos/sync"
        # Body and content-type pass through untouched.
        assert captured["body"] == body
        assert captured["headers"][b"content-type"] == (
            f"multipart/form-data; boundary={boundary}".encode()
        )

    def test_case2_json_with_route_converted(self):
        """application/json + route -> converted to multipart (the fix). This
        is the combination that previously returned 400."""
        captured = self._capture(
            headers=[
                (b"x-amzn-sagemaker-custom-attributes", b"route=/v1/videos/sync"),
                (b"content-type", b"application/json"),
            ],
            body=json.dumps(self.VIDEO_FIELDS).encode(),
        )
        assert captured["path"] == "/v1/videos/sync"
        ctype = captured["headers"][b"content-type"].decode()
        assert ctype.startswith("multipart/form-data; boundary=")
        fields = _parse_multipart(captured["body"], ctype)
        # int values arrive as form strings; FastAPI coerces them back to int.
        assert fields == {
            "prompt": "A cat playing with a ball",
            "num_frames": "17",
            "num_inference_steps": "30",
            "size": "480x320",
        }
