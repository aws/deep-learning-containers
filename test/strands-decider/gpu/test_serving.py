"""A strands-decider model served by the image answers the SageMaker contract on GPU.

The workflow starts the container as SageMaker does (`docker run IMAGE serve`, the model at /opt/ml/model) and
waits for /ping; these tests only call the server. ENDPOINT_URL is the server, EXPECTED_MODEL the checkpoint
folder served.
"""

import json
import os
import time
import urllib.error
import urllib.request

import pytest

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "http://localhost:8080")
EXPECTED_MODEL = os.environ["EXPECTED_MODEL"]

URGENT = "Help! My payouts have been failing for 3 days and nobody answers my emails!"
CALM = "Thanks, the order arrived on time and everything works as described."
DEPARTMENT = {
    "type": "choice",
    "instructions": "Route it.",
    "criteria": {"billing": "money, payouts, invoices", "technical": "bugs", "sales": "pricing"},
}
FRUSTRATION = {
    "type": "score",
    "instructions": "How frustrated is the customer?",
    "criteria": ["Calm", "Frustrated", "Very angry"],
}


def _post(path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{ENDPOINT_URL}{path}", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _ask(state: str, questions: dict) -> dict:
    status, body = _post("/invocations", {"state": state, "questions": questions})
    assert status == 200, body
    return body


def _noul(state: str) -> float:
    return _ask(state, {"q": {"type": "noul", "instructions": "Does this convey urgency?"}})[
        "answers"
    ]["q"]["noul"]


# First, while no other test has run: the warm-up at start must leave nothing for the first caller to compile.
# Without it, the first request on the Qwen3.5 models takes about 37 s (14 s for 3 questions). Warm, every
# request here takes 0.1 to 0.5 s on every model and GPU this runs on, so 3 s is a wide margin, not a latency
# target.
@pytest.mark.parametrize("n_questions", [1, 3, 17, 32])
def test_the_first_requests_after_ping_are_warm(n_questions):
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Is point {i} about money?"}
        for i in range(n_questions)
    }
    started = time.perf_counter()
    answers = _ask(f"A first request with {n_questions} questions.", questions)["answers"]
    elapsed = time.perf_counter() - started
    assert len(answers) == n_questions
    assert elapsed < 3, (
        f"{elapsed:.1f} s: a compile reached the request (warm-up missing or incomplete)"
    )


def test_ping_reports_the_served_model_on_cuda():
    with urllib.request.urlopen(f"{ENDPOINT_URL}/ping", timeout=30) as response:
        health = json.load(response)
    assert health["status"] == "ok"
    assert health["model"] == EXPECTED_MODEL
    assert health["device"].startswith("cuda")


def test_each_question_type_returns_a_distribution():
    answers = _ask(
        URGENT,
        {
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "department": DEPARTMENT,
            "frustration": FRUSTRATION,
        },
    )["answers"]

    assert 0.0 <= answers["is_urgent"]["noul"] <= 1.0

    department = answers["department"]
    assert department["choice"] in DEPARTMENT["criteria"]
    assert set(department["probabilities"]) == set(DEPARTMENT["criteria"])
    assert sum(department["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert department["choice"] == max(
        department["probabilities"], key=department["probabilities"].get
    )

    frustration = answers["frustration"]
    assert 0.0 <= frustration["score"] <= len(FRUSTRATION["criteria"]) - 1
    assert sum(frustration["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_the_model_tells_urgent_from_calm():
    # Far apart on every published model (an urgent payout failure reads 0.86 to 0.98), so the weights, the
    # adapter and the head all loaded: a model with random or missing weights can't separate them.
    assert _noul(URGENT) > 0.6
    assert _noul(CALM) < 0.4


def test_the_payout_failure_routes_to_billing_or_technical():
    answer = _ask(URGENT, {"department": DEPARTMENT})["answers"]["department"]
    assert answer["choice"] in ("billing", "technical")
    assert answer["probabilities"]["sales"] < 0.2


def test_invocations_and_v1_systemone_give_the_same_answers():
    body = {"state": URGENT, "questions": {"department": DEPARTMENT, "frustration": FRUSTRATION}}
    sagemaker_status, sagemaker = _post("/invocations", body)
    api_status, api = _post("/v1/systemone", body)
    assert (sagemaker_status, api_status) == (200, 200), (sagemaker, api)
    for name, answer in sagemaker["answers"].items():
        for option, p in answer["probabilities"].items():
            assert api["answers"][name]["probabilities"][option] == pytest.approx(p, abs=1e-3)


def test_one_question_answers_alone_as_among_others():
    # Several questions share one encoding of the state (the prefix cache); the answer must not depend on it.
    alone = _ask(URGENT, {"department": DEPARTMENT})["answers"]["department"]["probabilities"]
    together = _ask(URGENT, {"department": DEPARTMENT, "frustration": FRUSTRATION})["answers"][
        "department"
    ]
    for option, p in alone.items():
        assert together["probabilities"][option] == pytest.approx(p, abs=0.02)


def test_more_questions_than_one_batch_are_all_answered():
    # Above --max-batch (32) the engine runs the questions in chunks.
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Is point {i} about money?"} for i in range(40)
    }
    assert len(_ask(URGENT, questions)["answers"]) == 40


def test_a_malformed_question_is_rejected():
    status, _ = _post("/invocations", {"state": URGENT, "questions": {"q": {"type": "choice"}}})
    assert status == 422
