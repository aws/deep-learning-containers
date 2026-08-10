"""Tests for the CI-failure extraction in scripts/ci/autocurrency/agent-fix.py.

What this guards: everything extract_failure_info() returns is embedded verbatim
into the Bedrock prompt, so the size of that return value is a correctness
concern, not a cosmetic one. An unbounded job log overruns the model's context
window and the request is rejected outright.
"""

import io
import json
import zipfile


def _zip_with(entries: dict) -> bytes:
    """Build an in-memory log archive shaped like GitHub's run-logs zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeGitHub:
    """Stands in for urllib.request.urlopen, counting requests per URL."""

    def __init__(self, jobs_payload: dict, logs_zip: bytes):
        self.jobs_payload = jobs_payload
        self.logs_zip = logs_zip
        self.calls = []

    def __call__(self, req, *args, **kwargs):
        url = req.full_url
        self.calls.append(url)
        if url.endswith("/logs"):
            return _FakeResponse(self.logs_zip)
        return _FakeResponse(json.dumps(self.jobs_payload).encode())

    def log_downloads(self) -> int:
        return sum(1 for url in self.calls if url.endswith("/logs"))


def _job(name: str, step: str = "Run tests") -> dict:
    return {
        "name": name,
        "conclusion": "failure",
        "steps": [{"name": step, "conclusion": "failure"}],
    }


class TestTailLog:
    """_tail_log keeps the end of a log — that is where the failure is."""

    def test_short_log_is_returned_whole(self, agent_fix):
        lines = _tail_log_lines(agent_fix, ["line %d" % i for i in range(10)])
        assert lines == ["line %d" % i for i in range(10)]

    def test_long_log_is_clipped_to_max_lines(self, agent_fix):
        source = ["line %d" % i for i in range(5000)]
        lines = _tail_log_lines(agent_fix, source)

        # One marker line plus MAX_LOG_LINES of content
        assert len(lines) == agent_fix.MAX_LOG_LINES + 1
        assert lines[0] == f"... {5000 - agent_fix.MAX_LOG_LINES} earlier lines omitted ..."

    def test_clipped_log_keeps_the_tail_not_the_head(self, agent_fix):
        source = ["build chatter %d" % i for i in range(5000)]
        source[-1] = "ERROR: the actual failure"
        lines = _tail_log_lines(agent_fix, source)

        assert lines[-1] == "ERROR: the actual failure"
        assert "build chatter 0" not in lines

    def test_boundary_at_exactly_max_lines_is_not_marked_as_clipped(self, agent_fix):
        source = ["line %d" % i for i in range(agent_fix.MAX_LOG_LINES)]
        lines = _tail_log_lines(agent_fix, source)

        assert len(lines) == agent_fix.MAX_LOG_LINES
        assert not lines[0].startswith("...")


def _tail_log_lines(agent_fix, lines: list) -> list:
    return agent_fix._tail_log("\n".join(lines).encode())


class TestMatchTrackedJob:
    def test_matches_ignoring_separators_and_case(self, agent_fix):
        assert agent_fix._match_tracked_job("Security Test / ecr-scan") == "security-test"
        assert agent_fix._match_tracked_job("build-image (vllm)") == "build-image"

    def test_untracked_job_returns_none(self, agent_fix):
        assert agent_fix._match_tracked_job("pre-commit") is None


class TestExtractFailureInfo:
    def test_prompt_payload_is_bounded_by_max_log_lines(self, agent_fix, monkeypatch):
        """A huge job log must not reach the prompt untruncated (issue #3)."""
        huge_log = "\n".join("line %d" % i for i in range(200_000))
        fake = _FakeGitHub(
            jobs_payload={"jobs": [_job("build-image")]},
            logs_zip=_zip_with({"0_build-image.txt": huge_log}),
        )
        monkeypatch.setattr(agent_fix.urllib.request, "urlopen", fake)

        error_text, failed_jobs = agent_fix.extract_failure_info("123", "token", "owner/repo")

        assert failed_jobs == ["build-image"]
        # Header lines + the omission marker + at most MAX_LOG_LINES of log
        assert len(error_text.splitlines()) < agent_fix.MAX_LOG_LINES + 10
        assert "earlier lines omitted" in error_text
        # The tail survived
        assert "line 199999" in error_text

    def test_log_archive_is_downloaded_once_per_run(self, agent_fix, monkeypatch):
        """Three failed jobs in one run share one archive (issue #3)."""
        fake = _FakeGitHub(
            jobs_payload={
                "jobs": [_job("build-image"), _job("sanity-test"), _job("security-test")]
            },
            logs_zip=_zip_with(
                {
                    "0_build-image.txt": "boom",
                    "1_sanity-test.txt": "boom",
                    "2_security-test.txt": "boom",
                }
            ),
        )
        monkeypatch.setattr(agent_fix.urllib.request, "urlopen", fake)

        _, failed_jobs = agent_fix.extract_failure_info("123", "token", "owner/repo")

        assert sorted(failed_jobs) == ["build-image", "sanity-test", "security-test"]
        assert fake.log_downloads() == 1

    def test_untracked_and_successful_jobs_are_skipped(self, agent_fix, monkeypatch):
        jobs = {
            "jobs": [
                _job("pre-commit"),
                {"name": "build-image", "conclusion": "success", "steps": []},
                _job("sanity-test"),
            ]
        }
        fake = _FakeGitHub(jobs, _zip_with({"0_sanity-test.txt": "boom"}))
        monkeypatch.setattr(agent_fix.urllib.request, "urlopen", fake)

        error_text, failed_jobs = agent_fix.extract_failure_info("123", "token", "owner/repo")

        assert failed_jobs == ["sanity-test"]
        assert "pre-commit" not in error_text

    def test_log_download_failure_does_not_abort_extraction(self, agent_fix, monkeypatch):
        def exploding_urlopen(req, *args, **kwargs):
            if req.full_url.endswith("/logs"):
                raise OSError("503 from GitHub")
            return _FakeResponse(json.dumps({"jobs": [_job("build-image")]}).encode())

        monkeypatch.setattr(agent_fix.urllib.request, "urlopen", exploding_urlopen)

        error_text, failed_jobs = agent_fix.extract_failure_info("123", "token", "owner/repo")

        assert failed_jobs == ["build-image"]
        assert "Failed to download logs" in error_text
