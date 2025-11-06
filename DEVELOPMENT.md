# Developer Guide

This document describes how to set up a local development environment, run checks, and contribute changes to the project.

> 💡 All commands below assume a Unix-like environment (macOS or Linux).

---

## 1. Environment Setup

Create and activate a virtual environment using **[uv](https://docs.astral.sh/uv/)** — a fast Python package manager and virtualenv tool:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

---

## 2. Linting and Code Style

This project enforces linting and formatting through [pre-commit](https://pre-commit.com/#usage) hooks.

Install and configure:

```bash
uv pip install pre-commit
pre-commit install
```

To manually run all linters:

```bash
pre-commit run --all-files
```

Before committing or pushing changes, make sure your local Git and GitHub environments are properly configured.
Set your name and email — this is required for commit metadata and for the sign-off hook to work correctly.

```bash
git config --global user.name <YOUR NAME>
git config --global user.email <YOUR EMAIL>
```
