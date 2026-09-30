"""Unit tests for EFA version resolution in the build-image action.

A config may set ``efa_version: latest``. AWS publishes no version index for the
EFA installer, so the number is found by reading the install docs and proving the
answer against the bucket: ``aws-efa-installer-latest.tar.gz`` is the same object
as the newest numbered tarball, so their ETags match.

These tests stub the bucket and the docs page, so they exercise the decision
logic without touching the network.
"""

import pytest
import resolve_build_args as rba


def fake_bucket(published, latest):
    """Stub for the installer bucket: version -> ETag, None when not published."""
    etags = {version: f"etag-{version}" for version in published}
    etags["latest"] = f"etag-{latest}"
    return lambda version: etags.get(version)


class FakeResponse:
    """Minimal stand-in for the object urllib.request.urlopen returns."""

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_explicit_pin_is_returned_without_any_lookup(monkeypatch):
    """A pinned version must not depend on the network being up."""

    def explode(*_args):
        raise AssertionError("a pinned version must not trigger a lookup")

    monkeypatch.setattr(rba, "_efa_tarball_etag", explode)
    monkeypatch.setattr(rba, "_efa_version_from_docs", explode)

    assert rba.resolve_efa_version("1.47.0") == "1.47.0"


def test_docs_version_is_used_when_it_matches_latest(monkeypatch):
    monkeypatch.setattr(rba, "_efa_tarball_etag", fake_bucket(["1.50.0"], "1.50.0"))
    monkeypatch.setattr(rba, "_efa_version_from_docs", lambda: "1.50.0")

    assert rba.resolve_efa_version("latest") == "1.50.0"


def test_climbs_minor_releases_past_a_lagging_docs_page(monkeypatch):
    monkeypatch.setattr(
        rba, "_efa_tarball_etag", fake_bucket(["1.48.0", "1.49.0", "1.50.0"], "1.50.0")
    )
    monkeypatch.setattr(rba, "_efa_version_from_docs", lambda: "1.48.0")

    assert rba.resolve_efa_version("latest") == "1.50.0"


def test_climbs_a_patch_release(monkeypatch):
    monkeypatch.setattr(rba, "_efa_tarball_etag", fake_bucket(["1.50.0", "1.50.1"], "1.50.1"))
    monkeypatch.setattr(rba, "_efa_version_from_docs", lambda: "1.50.0")

    assert rba.resolve_efa_version("latest") == "1.50.1"


def test_unreadable_docs_page_falls_back_and_still_resolves(monkeypatch, capsys):
    """A moved docs page must not stall EFA currency or break the build."""
    monkeypatch.setattr(
        rba, "_efa_tarball_etag", fake_bucket([rba.EFA_VERSION_FALLBACK], rba.EFA_VERSION_FALLBACK)
    )
    monkeypatch.setattr(rba, "_efa_version_from_docs", lambda: None)

    assert rba.resolve_efa_version("latest") == rba.EFA_VERSION_FALLBACK
    # The warning is the only signal that the floor has become load-bearing.
    assert "Could not read an EFA version" in capsys.readouterr().out


def test_docs_naming_an_unpublished_version_falls_back(monkeypatch, capsys):
    """Docs can name a release before the tarball lands; never install a 404."""
    monkeypatch.setattr(
        rba, "_efa_tarball_etag", fake_bucket([rba.EFA_VERSION_FALLBACK], rba.EFA_VERSION_FALLBACK)
    )
    monkeypatch.setattr(rba, "_efa_version_from_docs", lambda: "9.99.0")

    assert rba.resolve_efa_version("latest") == rba.EFA_VERSION_FALLBACK
    # Distinct from an unreadable page: the page was fine, the tarball is not out.
    assert "no tarball is published for it yet" in capsys.readouterr().out


def test_fails_when_latest_cannot_be_reached(monkeypatch):
    monkeypatch.setattr(rba, "_efa_tarball_etag", lambda _version: None)
    monkeypatch.setattr(rba, "_efa_version_from_docs", lambda: "1.50.0")

    with pytest.raises(SystemExit) as excinfo:
        rba.resolve_efa_version("latest")
    assert excinfo.value.code == 1


def test_fails_when_no_published_version_matches_latest(monkeypatch):
    """`latest` exists but nothing numbered matches it, so nothing is provable."""
    monkeypatch.setattr(
        rba, "_efa_tarball_etag", lambda version: "etag-latest" if version == "latest" else None
    )
    monkeypatch.setattr(rba, "_efa_version_from_docs", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        rba.resolve_efa_version("latest")
    assert excinfo.value.code == 1


def test_fails_when_the_climb_budget_is_exhausted(monkeypatch):
    """A fallback too stale to reach `latest` must fail, not ship silently."""
    monkeypatch.setattr(rba, "EFA_MAX_PROBES", 2)
    monkeypatch.setattr(
        rba,
        "_efa_tarball_etag",
        lambda version: "etag-latest" if version == "latest" else "etag-older",
    )
    monkeypatch.setattr(rba, "_efa_version_from_docs", lambda: "1.20.0")

    with pytest.raises(SystemExit) as excinfo:
        rba.resolve_efa_version("latest")
    assert excinfo.value.code == 1


def test_successors_are_next_patch_then_next_minor():
    assert rba._successors("1.50.0") == ("1.50.1", "1.51.0")
    assert rba._successors("1.50.3") == ("1.50.4", "1.51.0")


def test_docs_parsing_compares_versions_numerically(monkeypatch):
    """1.9.0 must not outrank 1.50.0, which a string comparison would get wrong."""
    page = b"""
      curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.9.0.tar.gz
      curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz
    """
    monkeypatch.setattr(rba.urllib.request, "urlopen", lambda *a, **k: FakeResponse(page))

    assert rba._efa_version_from_docs() == "1.50.0"


def test_docs_parsing_returns_none_when_no_version_is_present(monkeypatch):
    monkeypatch.setattr(rba.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"<html/>"))

    assert rba._efa_version_from_docs() is None
