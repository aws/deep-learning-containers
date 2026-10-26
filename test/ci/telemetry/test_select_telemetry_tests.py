"""Unit tests for select_telemetry_tests.

A PR that only adds a REGION_MAPPING entry must run no image test, a PR that adds a
--framework choice must test the mock image as that framework only, and every other
telemetry change must test the mock image as MOCK_FRAMEWORK.
"""

import json
import subprocess
import textwrap

import pytest
import select_telemetry_tests as stt

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
                "base",
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
REMOVE_FRAMEWORK = ('            "vllm",\n', "")
LOGIC_CHANGE = ("TIMEOUT_SECS = 5", "TIMEOUT_SECS = 6")
MOCK = {stt.MOCK_FRAMEWORK}


def edit(source, *replacements):
    for old, new in replacements:
        assert old in source, f"test fixture does not contain {old!r}"
        source = source.replace(old, new)
    return source


def scope(base, head):
    return stt.classify(base, stt.parse_registry(head))


def test_unchanged_script_tests_nothing():
    assert not scope(SCRIPT, SCRIPT).frameworks


def test_new_region_tests_nothing():
    result = scope(SCRIPT, edit(SCRIPT, ADD_REGION))
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
    assert not scope(SCRIPT, head).frameworks


def test_new_framework_tests_the_mock_as_that_framework_only():
    assert scope(SCRIPT, edit(SCRIPT, ADD_FRAMEWORK)).frameworks == {"miles"}


def test_new_framework_and_region_tests_the_mock_as_that_framework_only():
    assert scope(SCRIPT, edit(SCRIPT, ADD_FRAMEWORK, ADD_REGION)).frameworks == {"miles"}


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param(('"us-west-2": "d8c0d063"', '"us-west-2": "00000000"'), id="changed-bucket"),
        pytest.param(('    "us-west-2": "d8c0d063",\n', ""), id="removed-region"),
        pytest.param(REMOVE_FRAMEWORK, id="removed-framework"),
        pytest.param(LOGIC_CHANGE, id="logic-change"),
        pytest.param(("required=True", "required=False"), id="framework-argument-change"),
    ],
)
def test_other_script_changes_test_the_mock_as_the_mock_framework(replacement):
    assert scope(SCRIPT, edit(SCRIPT, replacement)).frameworks == MOCK


def test_logic_change_with_new_framework_tests_both():
    assert scope(SCRIPT, edit(SCRIPT, LOGIC_CHANGE, ADD_FRAMEWORK)).frameworks == MOCK | {"miles"}


def test_removed_framework_is_reported():
    assert scope(SCRIPT, edit(SCRIPT, REMOVE_FRAMEWORK)).removed == {"vllm"}


def test_new_script_tests_the_mock_as_the_mock_framework():
    assert scope(None, SCRIPT).frameworks == MOCK


def test_unreadable_base_registry_tests_the_mock_as_the_mock_framework():
    base = edit(SCRIPT, ("REGION_MAPPING = {", "REGIONS = {"))
    assert scope(base, SCRIPT).frameworks == MOCK


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
    with pytest.raises(stt.RegistryError, match=message):
        stt.parse_registry(edit(SCRIPT, replacement))


# --------------------------------------------------------------------------- #
#                                   main()                                    #
# --------------------------------------------------------------------------- #

# config path -> metadata.framework
CONFIGS = {
    "base/cu130-runtime.yml": "base",
    "pytorch/ec2.yml": "pytorch",
    "vllm/ec2.yml": "vllm",
    "miles/ec2.yml": "miles",
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


def config(framework):
    return f"image:\n  name: {framework}-image\nmetadata:\n  framework: {framework}\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo whose main branch has the telemetry script, a test suite file, and configs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    files = {stt.DLC_SCRIPT: SCRIPT, "test/telemetry/test_telemetry.py": "def test_x(): pass\n"}
    for path, framework in CONFIGS.items():
        files[f"{stt.CONFIG_DIR}/{path}"] = config(framework)
    commit(repo, "base", files)
    git(repo, "checkout", "-q", "-b", "pr")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github_output"))
    return repo


def selected(repo, *args):
    """Run main() and return the frameworks to test the mock image as."""
    output = repo.parent / "github_output"
    output.write_text("")
    assert stt.main(list(args)) == 0
    (line,) = output.read_text().splitlines()
    key, value = line.split("=", 1)
    assert key == "frameworks"
    return json.loads(value)


def test_main_region_add_runs_no_image_test(repo):
    commit(repo, "add region", {stt.DLC_SCRIPT: edit(SCRIPT, ADD_REGION)})
    assert selected(repo, "--base-ref", "main") == []


def test_main_framework_add_tests_the_mock_as_that_framework(repo):
    commit(repo, "add miles", {stt.DLC_SCRIPT: edit(SCRIPT, ADD_FRAMEWORK, ADD_REGION)})
    assert selected(repo, "--base-ref", "main") == ["miles"]


def test_main_logic_change_tests_the_mock_as_the_mock_framework(repo):
    commit(repo, "logic", {stt.DLC_SCRIPT: edit(SCRIPT, LOGIC_CHANGE)})
    assert selected(repo, "--base-ref", "main") == [stt.MOCK_FRAMEWORK]


@pytest.mark.parametrize(
    "path",
    [
        "test/telemetry/test_telemetry.py",
        "scripts/docker/telemetry/bash_telemetry.sh.template",
        ".github/workflows/_reusable.telemetry-tests.yml",
        ".github/workflows/telemetry.pipeline.yml",
        "scripts/ci/telemetry_mock/Dockerfile",
    ],
)
def test_main_suite_change_tests_the_mock_as_the_mock_framework(repo, path):
    commit(repo, "suite", {path: "changed\n", stt.DLC_SCRIPT: edit(SCRIPT, ADD_REGION)})
    assert selected(repo, "--base-ref", "main") == [stt.MOCK_FRAMEWORK]


def test_main_suite_change_with_new_framework_tests_both(repo):
    files = {"test/telemetry/test_telemetry.py": "changed\n"}
    commit(repo, "suite", {**files, stt.DLC_SCRIPT: edit(SCRIPT, ADD_FRAMEWORK)})
    assert selected(repo, "--base-ref", "main") == sorted([stt.MOCK_FRAMEWORK, "miles"])


def test_main_selector_only_change_runs_no_image_test(repo):
    commit(repo, "selector", {"scripts/ci/select_telemetry_tests.py": "# changed\n"})
    assert selected(repo, "--base-ref", "main") == []


def test_main_ignores_base_branch_changes_on_merge_commit(repo):
    """CI checks out the PR merge commit; only the PR side of the merge may widen the scope."""
    commit(repo, "add region", {stt.DLC_SCRIPT: edit(SCRIPT, ADD_REGION)})
    git(repo, "checkout", "-q", "main")
    commit(
        repo,
        "unrelated suite change on main",
        {"test/telemetry/test_telemetry.py": "def test_y(): pass\n"},
    )
    git(repo, "merge", "-q", "--no-ff", "-m", "merge pr", "pr")
    assert selected(repo) == []


def test_main_warns_when_new_framework_has_no_config(repo, capsys):
    head = edit(SCRIPT, (ADD_FRAMEWORK[0], ADD_FRAMEWORK[0] + '            "milez",\n'))
    commit(repo, "typo", {stt.DLC_SCRIPT: head})
    assert selected(repo, "--base-ref", "main") == ["milez"]
    assert "::warning" in capsys.readouterr().out


def test_main_fails_when_a_removed_framework_is_still_in_use(repo, capsys):
    commit(repo, "drop vllm", {stt.DLC_SCRIPT: edit(SCRIPT, REMOVE_FRAMEWORK)})
    assert stt.main(["--base-ref", "main"]) == 1
    assert "vllm/ec2.yml" in capsys.readouterr().out


def test_main_allows_a_framework_rename_that_moves_its_configs(repo):
    head = edit(SCRIPT, ('            "vllm",\n', '            "vllm_server",\n'))
    files = {stt.DLC_SCRIPT: head, f"{stt.CONFIG_DIR}/vllm/ec2.yml": config("vllm_server")}
    commit(repo, "rename vllm", files)
    assert selected(repo, "--base-ref", "main") == sorted([stt.MOCK_FRAMEWORK, "vllm_server"])


def test_main_fails_when_the_mock_framework_is_not_registered(repo, capsys):
    head = edit(SCRIPT, ('            "base",\n', ""), LOGIC_CHANGE)
    files = {stt.DLC_SCRIPT: head, f"{stt.CONFIG_DIR}/base/cu130-runtime.yml": config("pytorch")}
    commit(repo, "drop base", files)
    assert stt.main(["--base-ref", "main"]) == 1
    assert "MOCK_FRAMEWORK" in capsys.readouterr().out


def test_main_fails_when_registry_is_unreadable(repo, capsys):
    commit(
        repo, "restructure", {stt.DLC_SCRIPT: edit(SCRIPT, ("REGION_MAPPING = {", "REGIONS = {"))}
    )
    assert stt.main(["--base-ref", "main"]) == 1
    assert "::error" in capsys.readouterr().out


def test_main_fails_when_base_ref_is_unknown(repo, capsys):
    assert stt.main(["--base-ref", "no-such-ref"]) == 1
    assert "::error" in capsys.readouterr().out
