#!/usr/bin/env python3
"""agent-fix.py — Diagnose CI failures and generate fixes using Bedrock Claude.

Called by the agent-currency-fix.yml workflow. Reads CI logs, assembles context,
calls Bedrock Claude Opus, and applies the returned file edits to the working tree.

Usage:
    python3 scripts/autocurrency/agent-fix.py \
        --logs-dir /tmp/ci-logs/ \
        --framework vllm \
        --branch auto-update/vllm-0.21.0
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
MAX_TOKENS = 8192
REGION = os.environ.get("AWS_REGION", "us-west-2")

# Max lines of log to include in context
MAX_LOG_LINES = 500


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
                # Include context: 2 lines before, the error, 2 lines after
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
    """Read a file and return its content, or empty string."""
    try:
        return Path(path).read_text()
    except (FileNotFoundError, PermissionError):
        return ""


def get_previous_fixes(branch: str) -> str:
    """Get commit messages of previous agent fixes on this branch."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "origin/main..HEAD", "--grep=[agent-fix]"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or "None"
    except subprocess.CalledProcessError:
        return "None"


def build_prompt(framework: str, branch: str, error_lines: str, dockerfile: str,
                 config_content: str, previous_fixes: str) -> str:
    """Assemble the prompt for Claude."""
    return f"""You are an automated CI fix agent for the AWS Deep Learning Containers repo.
A currency auto-update PR on branch `{branch}` has failed CI. Your job is to diagnose
the failure and produce minimal file edits to fix it.

## Rules
- ONLY fix the specific failure shown in the logs
- Do NOT delete or skip tests
- Do NOT modify files unrelated to the failure
- For CVE scan failures: either pin a safe version in the Dockerfile, or add to the allowlist if the package is vendored/unpatchable
- For "file not found" errors: search for the new path in the upstream repo
- For build errors: check if the upstream base image changed something (Python path, package layout, etc.)
- If the failure is TRANSIENT (capacity, timeout, runner crash): respond with {{"transient": true}}

## Context

### Framework: {framework}
### Branch: {branch}

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
{previous_fixes}

## Response Format

Respond with ONLY valid JSON. No markdown, no explanation outside the JSON.

If transient failure:
{{"transient": true, "reason": "brief explanation"}}

If fixable, return an array of edits:
{{
  "transient": false,
  "description": "one-line description of the fix for commit message",
  "edits": [
    {{"path": "relative/file/path", "action": "replace", "old": "exact string to find", "new": "replacement string"}},
    {{"path": "relative/file/path", "action": "create", "content": "full file content"}},
    {{"path": "relative/file/path", "action": "append_json", "entry": {{"vulnerability_id": "CVE-...", "reason": "..."}}}}
  ]
}}

If you cannot determine a fix:
{{"transient": false, "description": "unable to fix", "edits": []}}
"""


def call_bedrock(prompt: str) -> str:
    """Call Bedrock Claude and return the response text."""
    client = boto3.client("bedrock-runtime", region_name=REGION)
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def apply_edits(edits: list) -> list:
    """Apply file edits to the working tree. Returns list of modified files."""
    modified = []
    for edit in edits:
        path = edit["path"]
        action = edit["action"]

        if action == "replace":
            content = Path(path).read_text()
            old = edit["old"]
            if old not in content:
                print(f"WARNING: '{old[:80]}...' not found in {path}, skipping edit")
                continue
            content = content.replace(old, edit["new"])
            Path(path).write_text(content)
            modified.append(path)

        elif action == "create":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(edit["content"])
            modified.append(path)

        elif action == "append_json":
            content = json.loads(Path(path).read_text())
            content.append(edit["entry"])
            Path(path).write_text(json.dumps(content, indent=4) + "\n")
            modified.append(path)

        else:
            print(f"WARNING: Unknown action '{action}' for {path}, skipping")

    return modified


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

    # Build prompt and call LLM
    prompt = build_prompt(
        args.framework, args.branch, error_lines,
        dockerfile, config_content, previous_fixes,
    )

    print("Calling Bedrock Claude Opus 4.6...")
    response_text = call_bedrock(prompt)

    # Parse response
    try:
        # Strip markdown code fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse LLM response as JSON: {e}")
        print(f"Raw response:\n{response_text[:2000]}")
        sys.exit(1)

    # Handle transient failures
    if result.get("transient"):
        print(f"Transient failure detected: {result.get('reason', 'unknown')}")
        print("No code fix needed. Exiting.")
        sys.exit(0)

    # Handle no edits
    description = result.get("description", "automated fix")
    edits = result.get("edits", [])

    if not edits:
        print(f"Agent could not determine a fix: {description}")
        sys.exit(1)

    # Apply edits
    print(f"Applying {len(edits)} edit(s)...")
    modified = apply_edits(edits)

    if not modified:
        print("No files were actually modified. Exiting.")
        sys.exit(1)

    print(f"Modified files: {modified}")

    # Write description for the commit message
    Path("/tmp/agent-fix-description.txt").write_text(description)
    print(f"Fix description: {description}")
    print("Done.")


if __name__ == "__main__":
    main()
