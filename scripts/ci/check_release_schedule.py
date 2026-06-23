#!/usr/bin/env python3
"""Validate that autorelease workflow cron schedules match the release-schedule.yml data file.

Usage:
    python3 scripts/ci/check_release_schedule.py

Exit code: 0 if all schedules match, 1 if mismatches found.
"""

import sys
from pathlib import Path

import yaml

SCHEDULE_FILE = Path(".github/release-schedule.yml")
WORKFLOWS_DIR = Path(".github/workflows")


def extract_cron_from_workflow(workflow_path: Path) -> str | None:
    """Extract the cron expression from a workflow file."""
    with open(workflow_path) as f:
        content = yaml.safe_load(f)

    on = content.get("on") or content.get(True)
    if not on or not isinstance(on, dict):
        return None

    schedule = on.get("schedule")
    if not schedule or not isinstance(schedule, list):
        return None

    if len(schedule) == 0:
        return None

    return schedule[0].get("cron")


def main():
    if not SCHEDULE_FILE.exists():
        print(f"ERROR: Schedule file not found: {SCHEDULE_FILE}")
        return 1

    with open(SCHEDULE_FILE) as f:
        schedule_data = yaml.safe_load(f)

    schedules = schedule_data.get("schedules", [])
    if not schedules:
        print(f"ERROR: No schedules found in {SCHEDULE_FILE}")
        return 1

    errors = []
    checked = 0

    for entry in schedules:
        expected_cron = entry["cron"]
        workflow_name = entry["workflow"]
        workflow_path = WORKFLOWS_DIR / workflow_name

        if not workflow_path.exists():
            errors.append(f"  {workflow_name}: file not found")
            continue

        actual_cron = extract_cron_from_workflow(workflow_path)

        if actual_cron is None:
            errors.append(f"  {workflow_name}: no cron schedule found in workflow")
            continue

        if actual_cron != expected_cron:
            errors.append(
                f"  {workflow_name}: schedule mismatch\n"
                f"    release-schedule.yml: {expected_cron}\n"
                f"    workflow file:        {actual_cron}"
            )
            continue

        checked += 1

    # Check for autorelease workflows NOT listed in the schedule file
    scheduled_workflows = {entry["workflow"] for entry in schedules}
    autorelease_files = sorted(WORKFLOWS_DIR.glob("*.autorelease-*.yml"))

    for wf in autorelease_files:
        if wf.name not in scheduled_workflows:
            actual_cron = extract_cron_from_workflow(wf)
            if actual_cron:
                errors.append(
                    f"  {wf.name}: has cron '{actual_cron}' but is NOT listed in release-schedule.yml"
                )

    if errors:
        print(f"FAILED: Release schedule validation ({len(errors)} error(s))\n")
        for err in errors:
            print(err)
        print("\nUpdate .github/config/release-schedule.yml or fix the workflow cron.")
        return 1

    print(f"OK: All {checked} autorelease schedules match release-schedule.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
