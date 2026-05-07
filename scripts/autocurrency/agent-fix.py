#!/usr/bin/env python3
"""agent-fix.py — Diagnose CI failures and generate fixes using Bedrock Claude.

Called by the agent-currency-fix.yml workflow. Reads CI logs, assembles context,
calls Bedrock Claude Opus with a retry loop, and applies edits.

Edit format: Search/Replace blocks (industry standard used by Aider, Cline, Claude).
Matching: Exact first, then whitespace-normalized fuzzy fallback.
Retry: If edits fail to apply or response is unparseable, retries with error feedback.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Agent-driven CI fix for currency PRs")
    parser.add_argument("--logs-dir", required=True, help="Directory containing CI log files")
    parser.add_argument("--framework", required=True, help="Framework name (vllm, sglang)")
    parser.add_argument("--branch", required=True, help="PR branch name")
    return parser.parse_args()


def extract_error_lines(logs_dir: str) -> str:
    """Extract relevant error lines from CI logs."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return "No logs available."

    error_lines = []
    for log_file in sorted(logs_path.rglob("*.txt")):
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            lower = line.lower()
            if any(kw in lower for kw in ["error", "failed", "failure", "cve-", "not found", "exception", "denied"]):
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                chunk = lines[start:end]
                error_lines.append(f"--- {log_file.name}:{i+1} ---")
                error_lines.extend(chunk)
                error_lines.append("")

        if len(error_lines) > MAX_LOG_LINES:
            break

    if not error_lines:
        return "No error patterns found in logs."

    return "\n".join(error_lines[:MAX_LOG_LINES])


def read_file_if_exists(path: str) -> str:
    try:
        return Path(path).read_text()
    except (FileNotFoundError, PermissionError):
        return ""


def get_previous_fixes(branch: str) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "origin/main..HEAD", "--grep=[agent-fix]"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or "None"
    except subprocess.CalledProcessError:
        return "None"


# ---------------------------------------------------------------------------
# Search/Replace block parsing (industry standard format)
# ---------------------------------------------------------------------------

SEARCH_REPLACE_PATTERN = re.compile(
    r"^(.+?)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE$",
    re.MULTILINE | re.DOTALL,
)


def parse_search_replace_blocks(response: str) -> list:
    """Parse search/replace blocks from LLM response.

    Format:
        filepath
        <<<<<<< SEARCH
        exact text to find
        =======
        replacement text
        >>>>>>> REPLACE
    """
    blocks = []
    for match in SEARCH_REPLACE_PATTERN.finditer(response):
        filepath = match.group(1).strip().strip("`").strip()
        search = match.group(2)
        replace = match.group(3)
        blocks.append({"path": filepath, "search": search, "replace": replace})
    return blocks


def fuzzy_find(content: str, search: str) -> tuple:
    """Find the best match for search text in content.

    Returns (start_idx, end_idx) or (None, None) if no match.
    Strategy:
      1. Exact match
      2. Whitespace-normalized match (trailing spaces, line endings)

    If both fail, returns None — caller should retry with error feedback
    rather than guessing with fuzzy heuristics.
    """
    # Strategy 1: Exact match
    idx = content.find(search)
    if idx != -1:
        return idx, idx + len(search)

    # Strategy 2: Whitespace-normalized match
    def normalize_ws(s):
        return "\n".join(line.rstrip() for line in s.splitlines())

    norm_content = normalize_ws(content)
    norm_search = normalize_ws(search)
    idx = norm_content.find(norm_search)
    if idx != -1:
        # Map back to original content position by line number
        line_num = norm_content[:idx].count("\n")
        original_lines = content.splitlines(keepends=True)
        end_line = line_num + norm_search.count("\n")
        char_pos = sum(len(original_lines[i]) for i in range(line_num))
        char_end = sum(len(original_lines[i]) for i in range(end_line + 1))
        return char_pos, char_end

    return None, None


def apply_search_replace_blocks(blocks: list) -> tuple:
    """Apply search/replace blocks to files.

    Returns (modified_files, errors) where errors is a list of failure descriptions.
    """
    modified = []
    errors = []

    for block in blocks:
        path = block["path"]
        search = block["search"]
        replace = block["replace"]

        if not Path(path).exists():
            # "create" mode: if search is empty, create the file
            if not search.strip():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(replace)
                modified.append(path)
                continue
            errors.append(f"File not found: {path}")
            continue

        content = Path(path).read_text()
        start, end = fuzzy_find(content, search)

        if start is None:
            # Provide context for retry
            errors.append(
                f"SEARCH block not found in {path}.\n"
                f"  Searched for ({len(search)} chars): {search[:100]}...\n"
                f"  File content (first 500 chars): {content[:500]}"
            )
            continue

        # Apply replacement
        new_content = content[:start] + replace + content[end:]
        Path(path).write_text(new_content)
        modified.append(path)

    return modified, errors


# ---------------------------------------------------------------------------
# LLM interaction with retry loop
# ---------------------------------------------------------------------------

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

Otherwise, respond with search/replace blocks. For each file to modify, use this exact format:

<filepath>
<exact text to find in the file>

You may include multiple blocks for multiple files or multiple edits in one file.
Each SEARCH block must contain enough context to uniquely identify the location.
Include 1-2 surrounding lines for anchoring.

For JSON array files (like allowlists), show the SEARCH block as the last few lines
of the array, and the REPLACE block as those lines plus the new entry.

After all blocks, add a single line:
DESCRIPTION: <one-line commit message description>"""


def build_user_prompt(framework: str, branch: str, error_lines: str,
                      dockerfile: str, config_content: str,
                      previous_fixes: str, retry_context: str = "") -> str:
    prompt = f"""## Context
Framework: {framework}
Branch: {branch}

### CI Error Lines:
```
{error_lines}
```

### Current Dockerfile (docker/{framework}/Dockerfile):
```dockerfile
{dockerfile}
```

### Current Config (.github/config/image/{framework}-ec2.yml):
```yaml
{config_content}
```

### Previous agent fix attempts on this branch:
{previous_fixes}"""

    if retry_context:
        prompt += f"""

### RETRY — Previous attempt failed:
{retry_context}

Please fix the issues above and try again with corrected SEARCH blocks."""

    return prompt


def call_bedrock(system: str, user: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def main():
    args = parse_args()

    print(f"=== Currency Fix Agent ===")
    print(f"Framework: {args.framework}")
    print(f"Branch: {args.branch}")
    print()

    # Gather context
    error_lines = extract_error_lines(args.logs_dir)
    dockerfile = read_file_if_exists(f"docker/{args.framework}/Dockerfile")
    config_content = read_file_if_exists(f".github/config/image/{args.framework}-ec2.yml")
    previous_fixes = get_previous_fixes(args.branch)

    print(f"Error lines extracted: {len(error_lines.splitlines())} lines")
    print(f"Previous fix attempts: {previous_fixes}")
    print()

    # Retry loop
    retry_context = ""
    attempt = 0

    while attempt < MAX_LLM_RETRIES:
        attempt += 1
        print(f"--- LLM attempt {attempt}/{MAX_LLM_RETRIES} ---")

        user_prompt = build_user_prompt(
            args.framework, args.branch, error_lines,
            dockerfile, config_content, previous_fixes, retry_context,
        )

        print("Calling Bedrock Claude Opus 4.6...")
        response_text = call_bedrock(SYSTEM_PROMPT, user_prompt)

        # Check for transient failure
        if response_text.strip().startswith("TRANSIENT:"):
            reason = response_text.strip().split(":", 1)[1].strip()
            print(f"Transient failure detected: {reason}")
            print("No code fix needed. Exiting.")
            sys.exit(0)

        # Parse search/replace blocks
        blocks = parse_search_replace_blocks(response_text)

        if not blocks:
            # Could not parse any blocks — retry with feedback
            retry_context = (
                f"Could not parse any search/replace blocks from your response.\n"
                f"Your response started with: {response_text[:300]}...\n\n"
                f"Please respond using the exact format:\n"
                f"<filepath>\n<<<<<<< SEARCH\n<text>\n=======\n<replacement>\n>>>>>>> REPLACE"
            )
            print(f"No blocks parsed. Retrying with feedback...")
            continue

        print(f"Parsed {len(blocks)} edit block(s)")

        # Apply edits
        modified, errors = apply_search_replace_blocks(blocks)

        if errors:
            # Some or all edits failed — retry with error details
            retry_context = (
                f"{len(modified)} edit(s) applied, {len(errors)} failed:\n"
                + "\n".join(errors)
                + "\n\nPlease fix ONLY the failed SEARCH blocks. "
                + "Do NOT resend blocks that already applied successfully."
            )
            if modified:
                print(f"Partial apply: {len(modified)} succeeded, {len(errors)} failed. Retrying...")
            else:
                print(f"All edits failed. Retrying with error context...")
            continue

        # All edits applied successfully
        print(f"All {len(modified)} edit(s) applied successfully.")
        print(f"Modified files: {modified}")

        # Extract description
        desc_match = re.search(r"^DESCRIPTION:\s*(.+)$", response_text, re.MULTILINE)
        description = desc_match.group(1).strip() if desc_match else "automated fix"

        Path("/tmp/agent-fix-description.txt").write_text(description)
        print(f"Fix description: {description}")
        print("Done.")
        return

    # Exhausted retries
    print(f"ERROR: Failed to generate valid edits after {MAX_LLM_RETRIES} LLM attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
