#!/usr/bin/env python3
"""agent-fix.py — Diagnose CI failures on auto-update PRs using Bedrock Claude.

Uses search/replace blocks (Aider/Cline format) with retry-on-failure loop.
Called by agent-currency-fix.yml workflow.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import boto3

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
MAX_TOKENS = 16384
REGION = os.environ.get("AWS_REGION", "us-west-2")
MAX_LOG_LINES = 500
MAX_LLM_RETRIES = 3
CONTEXT_MAP_PATH = ".github/config/agent-context-files.yml"

SEARCH_REPLACE_PATTERN = re.compile(
    r"^([^\n]*?/[^\n]*)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE$",
    re.MULTILINE | re.DOTALL,
)

SYSTEM_PROMPT = """You are an automated CI fix agent for the AWS Deep Learning Containers repo.
A currency auto-update PR has failed CI. Diagnose the failure and produce minimal file edits.

## Rules
- ONLY fix the specific failure shown in the logs
- Do NOT delete or skip tests
- Do NOT modify files unrelated to the failure
- For CVE scan failures: pin a safe version in Dockerfile, or add to allowlist if vendored/unpatchable
- For "file not found" errors: find the new path in the upstream repo
- For build errors: check if upstream base image changed something

## Response Format

If the failure is TRANSIENT (capacity, timeout, runner crash), respond with exactly:
TRANSIENT: <brief reason>

Otherwise, respond with search/replace blocks:

<filepath>
<<<<<<< SEARCH
<exact text to find in the file>
=======
<replacement text>
>>>>>>> REPLACE

Include 1-2 surrounding lines in SEARCH for unique anchoring.
For JSON arrays (allowlists), SEARCH the last few lines and REPLACE with those lines plus the new entry.

End with: DESCRIPTION: <one-line commit message>"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--framework", required=True)
    p.add_argument("--branch", required=True)
    return p.parse_args()


def extract_error_lines(logs_dir: str) -> str:
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return "No logs available."

    error_lines = []
    keywords = ["error", "failed", "failure", "cve-", "not found", "exception", "denied"]

    for log_file in sorted(logs_path.rglob("*.txt")):
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in keywords):
                start, end = max(0, i - 2), min(len(lines), i + 3)
                error_lines.append(f"--- {log_file.name}:{i+1} ---")
                error_lines.extend(lines[start:end])
                error_lines.append("")
        if len(error_lines) > MAX_LOG_LINES:
            break

    return "\n".join(error_lines[:MAX_LOG_LINES]) or "No error patterns found in logs."


def read_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except (FileNotFoundError, PermissionError):
        return ""


def detect_failed_jobs(logs_dir: str) -> list:
    """Detect which CI jobs failed based on log filenames."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return []
    # Log files are named like "8_security-test _ ecr-vulnerability-scan.txt"
    job_names = set()
    for f in logs_path.rglob("*.txt"):
        name = f.stem.lower()
        for job in ["build-image", "sanity-test", "security-test", "telemetry-test", "upstream-tests", "sagemaker-test"]:
            if job in name:
                job_names.add(job)
    return list(job_names)


def load_context_files(framework: str, failed_jobs: list) -> dict:
    """Load relevant source files based on which jobs failed.

    Returns dict of {filepath: content}.
    """
    mapping_path = Path(CONTEXT_MAP_PATH)
    if not mapping_path.exists():
        return {p: read_file(p) for p in [
            f"docker/{framework}/Dockerfile",
            f".github/config/image/{framework}-ec2.yml",
            f"test/security/data/ecr_scan_allowlist/{framework}/framework_allowlist.json",
        ] if read_file(p)}

    # Parse YAML via subprocess (yq available on runners) or fallback to simple parsing
    try:
        import yaml
        config = yaml.safe_load(mapping_path.read_text())
    except ImportError:
        # Fallback: parse the simple YAML structure manually
        config = _parse_simple_yaml(mapping_path.read_text())

    paths = set()
    for p in config.get("common", []):
        paths.add(p.replace("{framework}", framework))

    jobs_map = config.get("jobs", {})
    for job in failed_jobs:
        for p in jobs_map.get(job, []):
            paths.add(p.replace("{framework}", framework))

    if not failed_jobs:
        for files in jobs_map.values():
            for p in files:
                paths.add(p.replace("{framework}", framework))

    return {p: content for p in sorted(paths) if (content := read_file(p))}


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML parser for our flat list-of-strings structure."""
    result = {"common": [], "jobs": {}}
    current_section = None
    current_job = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line == "common:":
            current_section = "common"
            current_job = None
        elif line == "jobs:":
            current_section = "jobs"
        elif current_section == "jobs" and line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            current_job = stripped.rstrip(":")
            result["jobs"][current_job] = []
        elif stripped.startswith("- "):
            value = stripped[2:].strip().strip('"')
            if current_section == "common":
                result["common"].append(value)
            elif current_job:
                result["jobs"][current_job].append(value)
    return result


def get_previous_fixes() -> str:
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", "origin/main..HEAD", "--grep=[agent-fix]"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip() or "None"
    except subprocess.CalledProcessError:
        return "None"


def parse_blocks(response: str) -> list:
    blocks = []
    for m in SEARCH_REPLACE_PATTERN.finditer(response):
        filepath = m.group(1).strip().strip("`").strip()
        # Strip common LLM artifacts: <filepath>, **filepath**, `filepath`
        filepath = re.sub(r"^<\w+>|<\/\w+>$", "", filepath).strip()
        filepath = filepath.strip("*").strip("`").strip()
        blocks.append({"path": filepath, "search": m.group(2), "replace": m.group(3)})
    return blocks


def find_match(content: str, search: str) -> tuple:
    """Exact match, then whitespace-normalized. Returns (start, end) or (None, None)."""
    idx = content.find(search)
    if idx != -1:
        return idx, idx + len(search)

    # Whitespace-normalized: strip trailing spaces per line
    norm = lambda s: "\n".join(line.rstrip() for line in s.splitlines())
    norm_content, norm_search = norm(content), norm(search)
    idx = norm_content.find(norm_search)
    if idx != -1:
        line_num = norm_content[:idx].count("\n")
        lines = content.splitlines(keepends=True)
        end_line = line_num + norm_search.count("\n")
        return sum(len(lines[i]) for i in range(line_num)), sum(len(lines[i]) for i in range(end_line + 1))

    return None, None


def apply_blocks(blocks: list) -> tuple:
    """Returns (modified_files, errors)."""
    modified, errors = [], []

    for b in blocks:
        path, search, replace = b["path"], b["search"], b["replace"]

        if not Path(path).exists():
            if not search.strip():  # Create new file
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(replace)
                modified.append(path)
            else:
                errors.append(f"File not found: {path}")
            continue

        content = Path(path).read_text()
        start, end = find_match(content, search)

        if start is None:
            errors.append(
                f"SEARCH not found in {path}.\n"
                f"  Searched for: {search[:100]}...\n"
                f"  Actual content (first 500 chars): {content[:500]}"
            )
            continue

        Path(path).write_text(content[:start] + replace + content[end:])
        modified.append(path)

    return modified, errors


def call_bedrock(system: str, user: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }),
    )
    return json.loads(resp["body"].read())["content"][0]["text"]


def build_prompt(framework, branch, error_lines, context_files,
                 previous_fixes, retry_context=""):
    files_section = ""
    for path, content in context_files.items():
        ext = Path(path).suffix.lstrip(".")
        lang = {"py": "python", "sh": "bash", "yml": "yaml", "json": "json"}.get(ext, "")
        files_section += f"\n### {path}:\n```{lang}\n{content}\n```\n"

    prompt = f"""## Context
Framework: {framework}
Branch: {branch}

### CI Error Lines:
```
{error_lines}
```
{files_section}
### Previous fix attempts on this branch:
{previous_fixes}"""

    if retry_context:
        prompt += f"\n\n### RETRY — Previous attempt failed:\n{retry_context}\n\nFix ONLY the failed SEARCH blocks. Do NOT resend already-applied blocks."
    return prompt


def main():
    args = parse_args()
    print(f"=== Currency Fix Agent: {args.framework} @ {args.branch} ===\n")

    error_lines = extract_error_lines(args.logs_dir)
    failed_jobs = detect_failed_jobs(args.logs_dir)
    context_files = load_context_files(args.framework, failed_jobs)
    previous_fixes = get_previous_fixes()

    # Debug: show what logs we have
    logs_path = Path(args.logs_dir)
    if logs_path.exists():
        log_files = list(logs_path.rglob("*.txt"))
        print(f"Log files found: {len(log_files)}")
        for f in log_files[:10]:
            print(f"  {f.name} ({f.stat().st_size} bytes)")
    else:
        print(f"WARNING: logs dir {args.logs_dir} does not exist!")

    print(f"Error lines extracted: {len(error_lines.splitlines())} lines")
    print(f"Error lines preview: {error_lines[:500]}")
    print(f"Failed jobs detected: {failed_jobs or 'none (including all files)'}")
    print(f"Context files loaded: {list(context_files.keys())}")
    print()

    retry_context = ""
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        print(f"--- Attempt {attempt}/{MAX_LLM_RETRIES} ---")

        prompt = build_prompt(args.framework, args.branch, error_lines,
                              context_files, previous_fixes, retry_context)
        print(f"Prompt size: {len(prompt)} chars")
        response = call_bedrock(SYSTEM_PROMPT, prompt)
        print(f"LLM response ({len(response)} chars):")
        print(response[:2000])
        if len(response) > 2000:
            print(f"  ... ({len(response) - 2000} more chars)")
        print()

        if response.strip().startswith("TRANSIENT:"):
            print(f"Transient: {response.strip().split(':', 1)[1].strip()}")
            sys.exit(0)

        blocks = parse_blocks(response)
        if blocks:
            print(f"Parsed {len(blocks)} block(s): {[b["path"] for b in blocks]}")
        if not blocks:
            retry_context = (
                f"Could not parse search/replace blocks from response.\n"
                f"Response started with: {response[:300]}...\n"
                f"Use exact format: <filepath>\\n<<<<<<< SEARCH\\n...\\n=======\\n...\\n>>>>>>> REPLACE"
            )
            print("No blocks parsed, retrying...")
            print(f"  Response preview: {response[:200]}")
            continue

        modified, errors = apply_blocks(blocks)
        if errors:
            retry_context = f"{len(modified)} applied, {len(errors)} failed:\n" + "\n".join(errors)
            print(f"{'Partial' if modified else 'All failed'}: {len(errors)} error(s), retrying...")
            for e in errors:
                print(f"  ERROR: {e[:300]}")
            continue

        # Success
        desc_match = re.search(r"^DESCRIPTION:\s*(.+)$", response, re.MULTILINE)
        description = desc_match.group(1).strip() if desc_match else "automated fix"
        Path("/tmp/agent-fix-description.txt").write_text(description)
        print(f"✅ {len(modified)} edit(s) applied: {modified}")
        print(f"Description: {description}")
        return

    print(f"ERROR: Failed after {MAX_LLM_RETRIES} attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
