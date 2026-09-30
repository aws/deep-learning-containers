#!/usr/bin/env python3
"""Fail a Dockerfile that fetches a third-party artifact without verifying it.

Two rules, applied only to sources outside Amazon's control:
  1. Every curl/wget must be verified in the same RUN, via `sha256sum -c`
     or `gpg --batch --verify`.
  2. Every remote image must be pinned by digest (`@sha256:...`).

Amazon-controlled sources (see AMAZON_HOST) are exempt: their integrity rests on
Amazon operating the channel, the same basis as the AL2023 dnf repos. Note this
keys on the distribution *host*, not on who authored the package — an AWS-owned
artifact served from github.com still needs a checksum, because GitHub is where
tampering would occur.

Usage:
    python scripts/ci/checks/check_pinned_downloads.py [dockerfile ...]

Defaults to docker/lambda/Dockerfile when no paths are given.
"""

import re
import sys
from pathlib import Path

DEFAULT_TARGETS = ["docker/lambda/Dockerfile"]

FETCH = re.compile(r"\b(?:curl|wget)\b")
VERIFY = re.compile(r"sha256sum\s+-c|sha512sum\s+-c|gpg\s+--batch\s+--verify")
# Fetching a signature or checksum file is itself the verification input.
FETCH_EXEMPT = re.compile(r"\.asc\b|\.sha256\b|\.sha512\b|\.sig\b|--recv-keys")
# Amazon-operated distribution channels, exempt from both rules.
AMAZON_HOST = re.compile(
    r"""(?x)
    (?://|@|^)(?:[\w.-]+\.)?           # optional subdomains
    (?: public\.ecr\.aws
      | [\w.-]*\.amazonaws\.com
      | [\w.-]*\.amazonlinux\.com
      | s3://
    )""",
    re.IGNORECASE,
)
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}")
# A digest supplied through an ARG, e.g. nvidia/cuda:13.0.3-runtime@${CUDA_DIGEST}.
DIGEST_ARG = re.compile(r"@\$\{?(\w+)\}?")
ARG_DEF = re.compile(r"^\s*ARG\s+(\w+)\s*=\s*(\S+)", re.IGNORECASE)

STAGE_AS = re.compile(r"^\s*FROM\s+(\S+)(?:\s+(?:as|AS)\s+(\S+))?", re.IGNORECASE)
COPY_FROM = re.compile(r"^\s*COPY\s+.*?--from=(\S+)", re.IGNORECASE)


def logical_instructions(text):
    """Yield (start_line, joined_text) per Dockerfile instruction, honoring backslashes."""
    lines = text.splitlines()
    buf, start = [], 1
    for lineno, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not buf:
            if not stripped or stripped.startswith("#"):
                continue
            start = lineno
        buf.append(raw)
        if not stripped.endswith("\\"):
            yield start, "\n".join(buf)
            buf = []
    if buf:
        yield start, "\n".join(buf)


def strip_comments(block):
    """Drop comment lines inside a multi-line instruction."""
    return "\n".join(ln for ln in block.splitlines() if not ln.strip().startswith("#"))


def check(path):
    text = Path(path).read_text()
    instructions = list(logical_instructions(text))

    # Local build stages are referenced by name, not pulled from a registry.
    stages = {
        m.group(2).lower() for _, b in instructions if (m := STAGE_AS.match(b)) and m.group(2)
    }
    stages.add("scratch")

    # ARG defaults, so a digest passed as @${VAR} can be resolved to a literal.
    arg_defaults = {}
    for _, block in instructions:
        if m := ARG_DEF.match(block.splitlines()[0]):
            arg_defaults.setdefault(m.group(1), m.group(2))

    problems = []

    for lineno, block in instructions:
        body = strip_comments(block)

        if FETCH.search(body) and not VERIFY.search(body):
            for offset, line in enumerate(body.splitlines()):
                if (
                    FETCH.search(line)
                    and not FETCH_EXEMPT.search(line)
                    and not AMAZON_HOST.search(line)
                ):
                    problems.append(
                        (
                            lineno + offset,
                            "unverified download (no sha256sum -c / gpg --verify "
                            f"in this RUN): {line.strip()[:90]}",
                        )
                    )

        refs = []
        if m := STAGE_AS.match(body):
            refs.append(m.group(1))
        if m := COPY_FROM.match(body.replace("\n", " ")):
            refs.append(m.group(1))

        for ref in refs:
            base = ref.split("@")[0].split(":")[0].lower()
            if base in stages or ref.startswith("$"):
                continue
            if AMAZON_HOST.search(ref):
                continue
            if DIGEST.search(ref):
                continue
            # @${VAR} counts as pinned only if VAR defaults to a literal digest.
            if m := DIGEST_ARG.search(ref):
                name = m.group(1)
                default = arg_defaults.get(name, "")
                if DIGEST.search("@" + default):
                    continue
                problems.append(
                    (
                        lineno,
                        f"digest ARG {name} has no literal sha256 default (got {default!r}): {ref}",
                    )
                )
                continue
            problems.append((lineno, f"image not pinned by digest: {ref}"))

    return problems


def main(argv):
    targets = argv or DEFAULT_TARGETS
    failed = False
    for target in targets:
        if not Path(target).is_file():
            print(f"ERROR: not a file: {target}", file=sys.stderr)
            failed = True
            continue
        problems = check(target)
        for lineno, msg in problems:
            print(f"ERROR: {target}:{lineno}: {msg}", file=sys.stderr)
        if problems:
            failed = True
        else:
            print(f"OK: {target} — all downloads verified, all images digest-pinned")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
