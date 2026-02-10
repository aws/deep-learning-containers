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
"""Tests for documentation link validation."""

import os
import re

import pytest

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")


@pytest.fixture(scope="module")
def markdown_files():
    """Get all markdown files in docs directory, excluding templates."""
    md_files = []
    for root, _, files in os.walk(DOCS_DIR):
        if "templates" in root.split(os.sep):
            continue
        for f in files:
            if f.endswith(".md") and not f.endswith(".template.md"):
                md_files.append(os.path.join(root, f))
    return md_files


def test_no_double_slashes_in_urls(markdown_files):
    """Test that URLs don't have double slashes (except in protocol)."""
    errors = []
    pattern = re.compile(r"\]\((https?://[^)]+)\)")

    for md_file in markdown_files:
        with open(md_file) as f:
            content = f.read()
        for match in pattern.finditer(content):
            url = match.group(1)
            path = re.sub(r"^https?://[^/]+", "", url)
            if "//" in path:
                rel_path = os.path.relpath(md_file, DOCS_DIR)
                errors.append(f"{rel_path}: {url}")

    error_msg = "\n".join(errors)
    assert not errors, f"URLs with double slashes found:\n{error_msg}"


def test_no_empty_links(markdown_files):
    """Test that there are no empty markdown links."""
    errors = []
    pattern = re.compile(r"\]\(\s*\)")

    for md_file in markdown_files:
        with open(md_file) as f:
            content = f.read()
        if pattern.search(content):
            rel_path = os.path.relpath(md_file, DOCS_DIR)
            errors.append(rel_path)

    error_msg = "\n".join(errors)
    assert not errors, f"Empty links found in:\n{error_msg}"


def test_internal_markdown_links_valid(markdown_files):
    """Test that internal .md links point to existing files."""
    errors = []
    link_pattern = re.compile(r"\]\(([^)h][^)]*\.md(?:#[^)]*)?)\)")

    for md_file in markdown_files:
        with open(md_file) as f:
            content = f.read()
        file_dir = os.path.dirname(md_file)

        for match in link_pattern.finditer(content):
            link = match.group(1).split("#")[0]
            target = os.path.normpath(os.path.join(file_dir, link))

            if not os.path.exists(target):
                rel_path = os.path.relpath(md_file, DOCS_DIR)
                errors.append(f"{rel_path} -> {link}")

    error_msg = "\n".join(errors)
    assert not errors, f"Broken internal links:\n{error_msg}"


def test_nav_yml_paths_exist():
    """Test that paths in .nav.yml files exist."""
    errors = []

    for root, _, files in os.walk(DOCS_DIR):
        for f in files:
            if f == ".nav.yml":
                nav_file = os.path.join(root, f)
                with open(nav_file) as nf:
                    content = nf.read()

                md_pattern = re.compile(r":\s*([^\s#]+\.md)")
                bare_pattern = re.compile(r"^\s*-\s+([^\s:]+\.md)", re.MULTILINE)

                for pattern in [md_pattern, bare_pattern]:
                    for match in pattern.finditer(content):
                        path = match.group(1)
                        target = os.path.join(root, path)

                        if not os.path.exists(target):
                            rel_nav = os.path.relpath(nav_file, DOCS_DIR)
                            errors.append(f"{rel_nav}: {path}")

    error_msg = "\n".join(errors)
    assert not errors, f"Missing nav paths:\n{error_msg}"


def test_internal_directory_links_valid(markdown_files):
    """Test that internal directory-style links (trailing slash) resolve to existing paths."""
    errors = []
    # Match relative links with trailing slash: ](path/) or href="path/"
    md_pattern = re.compile(r"\]\(([^)h][^)]*?/)\)")
    href_pattern = re.compile(r'href="([^"h][^"]*?/)"')

    for md_file in markdown_files:
        with open(md_file) as f:
            content = f.read()
        file_dir = os.path.dirname(md_file)

        for pattern in [md_pattern, href_pattern]:
            for match in pattern.finditer(content):
                link = match.group(1)
                target_dir = os.path.normpath(os.path.join(file_dir, link))
                target_index = os.path.join(target_dir, "index.md")
                # A trailing-slash link is valid if the directory has index.md
                # or if a .md file exists with the same name (MkDocs resolves both)
                target_md = os.path.normpath(os.path.join(file_dir, link.rstrip("/"))) + ".md"

                if not (os.path.exists(target_index) or os.path.exists(target_md)):
                    rel_path = os.path.relpath(md_file, DOCS_DIR)
                    errors.append(f"{rel_path} -> {link}")

    error_msg = "\n".join(errors)
    assert not errors, f"Broken internal directory links:\n{error_msg}"


def test_no_trailing_spaces_in_urls(markdown_files):
    """Test that URLs don't have trailing spaces before closing paren."""
    errors = []
    pattern = re.compile(r"\]\(https?://[^\)]*\s+\)")

    for md_file in markdown_files:
        with open(md_file) as f:
            content = f.read()
        if pattern.search(content):
            rel_path = os.path.relpath(md_file, DOCS_DIR)
            errors.append(rel_path)

    error_msg = "\n".join(errors)
    assert not errors, f"URLs with trailing spaces in:\n{error_msg}"
