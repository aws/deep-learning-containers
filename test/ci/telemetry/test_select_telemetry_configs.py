"""Unit tests for select_telemetry_configs.

A PR that only adds a REGION_MAPPING entry must test no images, a PR that adds a
--framework choice must test only that framework's images, and every other
telemetry change must test every released image of a registered framework.
"""

import json
import subprocess
import textwrap

import pytest
import select_telemetry_configs as sts

SCRIPT = textwrap.dedent(
    """\
    import argparse

    TIMEOUT_SECS = 5
    REGION_MAPPING = {
        "us-east-1": "487d6534",
        "us-west-2": "d8c0d063",
    }


    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--framework",
            choices=[
                "pytorch",
                "vllm",
            ],
            required=True,
        )
        return parser.parse_known_args()
    """
)

ADD_REGION = (
    '    "us-west-2": "d8c0d063",\n',
    '    "us-west-2": "d8c0d063",\n    "me-west-1": "1b9ff200",\n',
)
ADD_FRAMEWORK = ('            "vllm",\n', '            "vllm",\n            "miles",\n')


def edit(source, *replacements):
    for old, new in replacements:
        assert old in source, f"test fixture does not contain {old!r}"
        source = source.replace(old, new)
    return source


def scope(base, head):
    return sts.classify(base, sts.parse_registry(head))


def test_unchanged_script_tests_nothing():
    result = scope(SCRIPT, SCRIPT)
    assert not result.run_all
    assert not result.frameworks


def test_new_region_tests_nothing():
    result = scope(SCRIPT, edit(SCRIPT, ADD_REGION))
    assert not result.run_all
    assert not result.frameworks
    assert "me-west-1" in result.reason


def test_comment_and_reorder_tests_nothing():
    head = edit(
        SCRIPT,
        (
            '    "us-east-1": "487d6534",\n    "us-west-2": "d8c0d063",\n',
            '    # sorted\n    "us-west-2": "d8c0d063",\n    "us-east-1": "487d6534",\n',
        ),
        (
            '            "pytorch",\n            "vllm",\n',
            '            "vllm",\n            "pytorch",\n',
        ),
    )
    result = scope(SCRIPT, head)
    assert not result.run_all
    assert not result.frameworks


def test_new_framework_tests_only_that_framework():
    result = scope(SCRIPT, edit(SCRIPT, ADD_FRAMEWORK))
    assert not result.run_all
    assert result.frameworks == {"miles"}


def test_new_framework_and_region_tests_only_that_framework():
    result = scope(SCRIPT, edit(SCRIPT, ADD_FRAMEWORK, ADD_REGION))
    assert not result.run_all
    assert result.frameworks == {"miles"}


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param(('"us-west-2": "d8c0d063"', '"us-west-2": "00000000"'), id="changed-bucket"),
        pytest.param(('    "us-west-2": "d8c0d063",\n', ""), id="removed-region"),
        pytest.param(('            "vllm",\n', ""), id="removed-framework"),
        pytest.param(("TIMEOUT_SECS = 5", "TIMEOUT_SECS = 6"), id="logic-change"),
        pytest.param(("required=True", "required=False"), id="framework-argument-change"),
    ],
)
def test_other_script_changes_test_everything(replacement):
    assert scope(SCRIPT, edit(SCRIPT, replacement)).run_all


def test_new_script_tests_everything():
    assert scope(None, SCRIPT).run_all


def test_unreadable_base_registry_tests_everything():
    base = edit(SCRIPT, ("REGION_MAPPING = {", "REGIONS = {"))
    assert scope(base, SCRIPT).run_all


@pytest.mark.parametrize(
    "replacement, message",
    [
        pytest.param(("REGION_MAPPING = {", "REGIONS = {"), "REGION_MAPPING", id="no-regions"),
        pytest.param(('"--framework"', '"--fw"'), "--framework", id="no-framework-argument"),
        pytest.param(("        choices=[", "        nargs=["), "choices", id="no-choices"),
        pytest.param(('"487d6534"', "BUCKET"), "REGION_MAPPING", id="non-literal-region"),
        pytest.param(ADD_REGION[:1] + (ADD_REGION[0] * 2,), "repeats", id="duplicate-region"),
        pytest.param(
            ADD_FRAMEWORK[:1] + (ADD_FRAMEWORK[0] * 2,), "repeats", id="duplicate-framework"
        ),
    ],
)
def test_parse_registry_rejects_unexpected_registry(replacement, message):
    with pytest.raises(sts.RegistryError, match=message):
        sts.parse_registry(edit(SCRIPT, replacement))


# --------------------------------------------------------------------------- #
#                         Config selection and main()                         #
# --------------------------------------------------------------------------- #

CONFIGS = {
    "pytorch/ec2.yml": ("pytorch-ec2", "pytorch", True),
    "vllm/ec2.yml": ("vllm-ec2", "vllm", True),
    "vllm/hyperpod.yml": ("vllm-hyperpod", "vllm", False),
    "miles/ec2.yml": ("miles-ec2", "miles", True),
    "sklearn/sagemaker.yml": ("sklearn-sagemaker", "sklearn", True),
}


def git(repo, *args):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=ci",
            "-c",
            "user.email=ci@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write(repo, path, text):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def commit(repo, message, files):
    for path, text in files.items():
        write(repo, path, text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo whose main branch has the telemetry script, a test suite file, and configs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    files = {sts.DLC_SCRIPT: SCRIPT, "test/telemetry/test_telemetry.py": "def test_x(): pass\n"}
    for path, (name, framework, release) in CONFIGS.items():
        files[f"{sts.CONFIG_DIR}/{path}"] = (
            f"image:\n  name: {name}\nmetadata:\n  framework: {framework}\n"
            f"release:\n  release: {str(release).lower()}\n"
        )
    commit(repo, "base", files)
    git(repo, "checkout", "-q", "-b", "pr")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github_output"))
    return repo


def selected(repo, *args):
    """Run main() and return the selected config names."""
    output = repo.parent / "github_output"
    output.write_text("")
    assert sts.main(list(args)) == 0
    (line,) = output.read_text().splitlines()
    key, value = line.split("=", 1)
    assert key == "configs"
    return sorted(c["name"] for c in json.loads(value))


def test_main_region_add_selects_no_configs(repo):
    commit(repo, "add region", {sts.DLC_SCRIPT: edit(SCRIPT, ADD_REGION)})
    assert selected(repo, "--base-ref", "main") == []


def test_main_framework_add_selects_only_its_configs(repo):
    commit(repo, "add miles", {sts.DLC_SCRIPT: edit(SCRIPT, ADD_FRAMEWORK, ADD_REGION)})
    assert selected(repo, "--base-ref", "main") == ["miles-ec2"]


def test_main_logic_change_selects_released_configs_of_registered_frameworks(repo):
    commit(repo, "logic", {sts.DLC_SCRIPT: edit(SCRIPT, ("TIMEOUT_SECS = 5", "TIMEOUT_SECS = 6"))})
    # vllm-hyperpod is unreleased; miles and sklearn are not registered
    assert selected(repo, "--base-ref", "main") == ["pytorch-ec2", "vllm-ec2"]


@pytest.mark.parametrize(
    "path",
    [
        "test/telemetry/test_telemetry.py",
        "scripts/docker/telemetry/bash_telemetry.sh.template",
        ".github/workflows/_reusable.telemetry-tests.yml",
        ".github/workflows/telemetry.pipeline.yml",
    ],
)
def test_main_suite_change_selects_everything(repo, path):
    commit(repo, "suite", {path: "changed\n", sts.DLC_SCRIPT: edit(SCRIPT, ADD_REGION)})
    assert selected(repo, "--base-ref", "main") == ["pytorch-ec2", "vllm-ec2"]


def test_main_selector_only_change_selects_no_configs(repo):
    commit(repo, "selector", {"scripts/ci/select_telemetry_configs.py": "# changed\n"})
    assert selected(repo, "--base-ref", "main") == []


def test_main_ignores_base_branch_changes_on_merge_commit(repo):
    """CI checks out the PR merge commit; only the PR side of the merge may widen the scope."""
    commit(repo, "add region", {sts.DLC_SCRIPT: edit(SCRIPT, ADD_REGION)})
    git(repo, "checkout", "-q", "main")
    commit(
        repo,
        "unrelated suite change on main",
        {"test/telemetry/test_telemetry.py": "def test_y(): pass\n"},
    )
    git(repo, "merge", "-q", "--no-ff", "-m", "merge pr", "pr")
    assert selected(repo) == []


def test_main_warns_when_new_framework_has_no_released_config(repo, capsys):
    head = edit(SCRIPT, (ADD_FRAMEWORK[0], ADD_FRAMEWORK[0] + '            "milez",\n'))
    commit(repo, "typo", {sts.DLC_SCRIPT: head})
    assert selected(repo, "--base-ref", "main") == []
    assert "::warning" in capsys.readouterr().out


def test_main_fails_when_registry_is_unreadable(repo, capsys):
    commit(
        repo, "restructure", {sts.DLC_SCRIPT: edit(SCRIPT, ("REGION_MAPPING = {", "REGIONS = {"))}
    )
    assert sts.main(["--base-ref", "main"]) == 1
    assert "::error" in capsys.readouterr().out


def test_main_fails_when_base_ref_is_unknown(repo, capsys):
    assert sts.main(["--base-ref", "no-such-ref"]) == 1
    assert "::error" in capsys.readouterr().out
