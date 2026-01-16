# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Documentation generation from release configs."""

from collections import defaultdict
from datetime import date
from pathlib import Path

from config_loader import load_all_configs
from constants import (
    ALLOWED_SECTIONS,
    FRAMEWORK_NAMES,
    FRAMEWORK_ORDER,
    REPOSITORY_NAMES,
    REPOSITORY_ORDER,
    SECTION_TITLES,
)
from jinja2 import Environment, FileSystemLoader

SRC_DIR = Path(__file__).parent
DOCS_DIR = SRC_DIR.parent
TEMPLATES_DIR = SRC_DIR / "templates"
REFERENCE_DIR = DOCS_DIR / "reference"
RELEASENOTES_DIR = DOCS_DIR / "releasenotes"

env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


def _platform_title(platform: str) -> str:
    """Convert platform to display title."""
    return {
        "sagemaker": "Amazon SageMaker",
        "ec2": "EC2, ECS, and EKS",
    }.get(platform, platform)


def _os_title(os_str: str) -> str:
    """Convert OS string to display title."""
    if os_str.startswith("ubuntu"):
        version = os_str.replace("ubuntu", "")
        return f"Ubuntu {version}"
    return os_str


def _sort_key(item: dict) -> tuple:
    """Sort key: version desc, platform (sagemaker first), accelerator (gpu first)."""
    version = -float(item["version"].replace(".", ""))
    platform_order = {"Amazon SageMaker": 0, "sagemaker": 0, "EC2, ECS, and EKS": 1, "ec2": 1}.get(
        item.get("platform", ""), 2
    )
    accel = item.get("accelerator", "").upper()
    accel_order = {"GPU": 0, "CPU": 1}.get(accel, 2)
    return (version, platform_order, accel_order)


def generate_support_policy(configs: list[dict], dry_run: bool = False) -> str:
    """Generate support_policy.md from configs."""
    today = date.today()
    supported, unsupported = [], []

    # Group by framework+version to deduplicate
    seen = set()
    for cfg in configs:
        meta = cfg["metadata"]
        key = (meta["framework"], meta["version"])
        if key in seen:
            continue
        seen.add(key)

        row = {
            "framework": FRAMEWORK_NAMES.get(meta["framework"], meta["framework"]),
            "version": meta["version"],
            "ga_date": str(meta["ga_date"]),
            "eop_date": str(meta["eop_date"]),
        }
        if meta["eop_date"] >= today:
            supported.append(row)
        else:
            unsupported.append(row)

    # Sort by framework order, then version descending
    def sort_key(row):
        fw = row["framework"].lower()
        order = FRAMEWORK_ORDER.index(fw) if fw in FRAMEWORK_ORDER else 999
        return (order, -float(row["version"].replace(".", "")))

    supported.sort(key=sort_key)
    unsupported.sort(key=sort_key)

    template = env.get_template("support_policy.template.md")
    content = template.render(supported=supported, unsupported=unsupported)

    if not dry_run:
        (REFERENCE_DIR / "support_policy.md").write_text(content)
    return content


def generate_available_images(configs: list[dict], dry_run: bool = False) -> str:
    """Generate available_images.md from configs."""
    # Group configs by repository
    by_repo = defaultdict(list)
    for cfg in configs:
        by_repo[cfg["repository"]].append(cfg)

    repositories = []
    for repo_name in REPOSITORY_ORDER:
        if repo_name not in by_repo:
            continue
        repo_configs = by_repo[repo_name]
        images = []
        for cfg in repo_configs:
            meta = cfg["metadata"]
            packages = cfg.get("packages", {})
            images.append(
                {
                    "framework": FRAMEWORK_NAMES.get(meta["framework"], meta["framework"]),
                    "version": meta["version"],
                    "python": cfg["environment"]["python"],
                    "accelerator": meta["accelerator"].upper(),
                    "cuda": packages.get("cuda", "-"),
                    "platform": _platform_title(meta["platform"]),
                    "tag": cfg["image"]["tags"][0],
                }
            )
        # Sort by version desc, platform (sagemaker first), accelerator (gpu first)
        images.sort(key=_sort_key)

        public_registry = any(c["image"].get("public_registry") for c in repo_configs)
        repositories.append(
            {
                "name": repo_name,
                "display_name": REPOSITORY_NAMES.get(repo_name, repo_name),
                "public_registry": public_registry,
                "images": images,
            }
        )

    template = env.get_template("available_images.template.md")
    content = template.render(repositories=repositories)

    if not dry_run:
        (REFERENCE_DIR / "available_images.md").write_text(content)
    return content


def generate_release_notes(configs: list[dict], dry_run: bool = False) -> list[str]:
    """Generate release notes for each config."""
    template = env.get_template("release_notes.template.md")
    index_template = env.get_template("release_notes_index.template.md")
    generated = []

    # Group configs by framework for index generation
    by_framework = defaultdict(list)

    for cfg in configs:
        meta = cfg["metadata"]
        framework = meta["framework"]
        version = meta["version"]
        accelerator = meta["accelerator"]
        platform = meta["platform"]
        arch = meta.get("architecture", "x86_64")
        job_type = meta["job_type"]

        # Build output filename
        arch_suffix = "-arm64" if arch == "arm64" else ""
        filename = f"{version}-{accelerator}-{job_type}{arch_suffix}-{platform}.md"
        output_dir = RELEASENOTES_DIR / framework
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        content = template.render(
            framework_name=FRAMEWORK_NAMES.get(framework, framework),
            version=version,
            job_type_title=job_type.title(),
            platform_title=_platform_title(platform),
            os_title=_os_title(cfg["environment"]["os"]),
            python=cfg["environment"]["python"],
            architecture=arch,
            packages=cfg.get("packages", {}),
            tags=cfg["image"]["tags"],
            repository=cfg["repository"],
            sections=cfg.get("sections", {}),
            sections_config={k: SECTION_TITLES[k] for k in ALLOWED_SECTIONS},
        )

        if not dry_run:
            output_path.write_text(content)
        generated.append(str(output_path))

        # Collect for index
        by_framework[framework].append(
            {
                "filename": filename,
                "version": version,
                "accelerator": accelerator.upper(),
                "job_type": job_type.title(),
                "platform": _platform_title(platform),
                "arch": arch,
                "eop_date": meta["eop_date"],
            }
        )

    # Generate index.md for each framework
    for framework, releases in by_framework.items():
        # Sort by version desc, platform (sagemaker first), accelerator (gpu first)
        releases.sort(key=_sort_key)

        # Split into supported and deprecated based on EOP date
        today = date.today()
        supported = [r for r in releases if r["eop_date"] >= today]
        deprecated = [r for r in releases if r["eop_date"] < today]

        output_dir = RELEASENOTES_DIR / framework
        index_path = output_dir / "index.md"

        index_content = index_template.render(
            framework_name=FRAMEWORK_NAMES.get(framework, framework),
            supported=supported,
            deprecated=deprecated,
        )

        if not dry_run:
            index_path.write_text(index_content)

    return generated


def generate_all(dry_run: bool = False) -> None:
    """Generate all documentation files."""
    configs = load_all_configs()
    print(f"Loaded {len(configs)} configs")

    generate_support_policy(configs, dry_run)
    print("Generated support_policy.md")

    generate_available_images(configs, dry_run)
    print("Generated available_images.md")

    release_notes = generate_release_notes(configs, dry_run)
    print(f"Generated {len(release_notes)} release notes")


if __name__ == "__main__":
    generate_all()
