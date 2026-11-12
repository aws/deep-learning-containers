from datetime import datetime, timedelta, timezone
from unittest import mock

import boto3
import pytest
import sagemaker
from sagemaker.instance_group import InstanceGroup

import test.sagemaker_tests as sagemaker_tests
from test.sagemaker_tests import TrainingJobCleanupSession, get_instance_type_fallbacks


class FakeSageMakerClient:
    """Returns the given describe_training_job responses in order, repeating the last one"""

    def __init__(self, responses):
        self.responses = responses
        self.describe_calls = 0
        self.stopped = []

    def describe_training_job(self, TrainingJobName):
        response = self.responses[min(self.describe_calls, len(self.responses) - 1)]
        self.describe_calls += 1
        return response

    def stop_training_job(self, TrainingJobName):
        self.stopped.append(TrainingJobName)


def _job(secondary_status, pending_minutes_ago=None, status="InProgress"):
    now = datetime.now(timezone.utc)
    transitions = [{"Status": "Starting", "StartTime": now - timedelta(minutes=30)}]
    if pending_minutes_ago is not None:
        transitions.append(
            {"Status": "Pending", "StartTime": now - timedelta(minutes=pending_minutes_ago)}
        )
    if secondary_status not in ("Starting", "Pending"):
        transitions.append({"Status": secondary_status, "StartTime": now})
    elif secondary_status == "Starting" and pending_minutes_ago is not None:
        transitions.append({"Status": "Starting", "StartTime": now - timedelta(minutes=1)})
    return {
        "TrainingJobStatus": status,
        "SecondaryStatus": secondary_status,
        "SecondaryStatusTransitions": transitions,
    }


def _session(responses, pending_timeout_minutes=10):
    session = TrainingJobCleanupSession(boto_session=boto3.Session(region_name="us-west-2"))
    session.sagemaker_client = FakeSageMakerClient(responses)
    session.pending_timeout_minutes = pending_timeout_minutes
    return session


@pytest.fixture(autouse=True)
def no_sleep():
    with mock.patch.object(sagemaker_tests.time, "sleep"):
        yield


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker_training_job_cleanup")
def test_job_pending_past_the_limit_is_stopped():
    session = _session([_job("Pending", pending_minutes_ago=20)])
    with mock.patch.object(sagemaker.Session, "logs_for_job") as sdk_wait:
        with pytest.raises(sagemaker.exceptions.CapacityError, match="CapacityError"):
            session.logs_for_job("job", wait=True)
    sdk_wait.assert_not_called()
    assert session.sagemaker_client.stopped == ["job"]


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker_training_job_cleanup")
def test_job_back_to_starting_after_long_pending_is_not_stopped():
    # The job waited 12 minutes for capacity, got it, and is starting its instances
    session = _session([_job("Starting", pending_minutes_ago=12), _job("Downloading")])
    with mock.patch.object(sagemaker.Session, "logs_for_job") as sdk_wait:
        session.logs_for_job("job", wait=True)
    sdk_wait.assert_called_once()
    assert session.sagemaker_client.stopped == []


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker_training_job_cleanup")
def test_short_pending_keeps_waiting():
    session = _session(
        [_job("Pending", pending_minutes_ago=2), _job("Pending", pending_minutes_ago=3)]
        + [_job("Downloading")]
    )
    with mock.patch.object(sagemaker.Session, "wait_for_job") as sdk_wait:
        session.wait_for_job("job")
    sdk_wait.assert_called_once()
    assert session.sagemaker_client.stopped == []


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker_training_job_cleanup")
def test_no_limit_waits_like_the_sdk():
    session = _session([_job("Pending", pending_minutes_ago=99)], pending_timeout_minutes=None)
    with mock.patch.object(sagemaker.Session, "logs_for_job") as sdk_wait:
        session.logs_for_job("job", wait=True)
    sdk_wait.assert_called_once()
    assert session.sagemaker_client.describe_calls == 0


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker_training_job_cleanup")
def test_interrupted_wait_stops_the_job():
    session = _session([_job("Training")], pending_timeout_minutes=None)
    with mock.patch.object(sagemaker.Session, "logs_for_job", side_effect=TimeoutError):
        with pytest.raises(TimeoutError):
            session.logs_for_job("job", wait=True)
    assert session.sagemaker_client.stopped == ["job"]


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker_training_job_cleanup")
def test_failed_job_is_not_stopped():
    session = _session([_job("Failed", status="Failed")], pending_timeout_minutes=None)
    error = sagemaker.exceptions.UnexpectedStatusException(
        message="AlgorithmError", allowed_statuses=["Completed"], actual_status="Failed"
    )
    with mock.patch.object(sagemaker.Session, "wait_for_job", side_effect=error):
        with pytest.raises(sagemaker.exceptions.UnexpectedStatusException):
            session.wait_for_job("job")
    assert session.sagemaker_client.stopped == []


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker_training_job_cleanup")
def test_instance_type_fallbacks():
    single = get_instance_type_fallbacks({"instance_type": "ml.g6.xlarge", "instance_count": 3})
    assert [args["instance_type"] for args in single] == [
        "ml.g6.xlarge",
        "ml.g6e.xlarge",
        "ml.g5.xlarge",
        "ml.g4dn.xlarge",
    ]
    assert all(args["instance_count"] == 3 for args in single)

    groups = get_instance_type_fallbacks(
        {"instance_groups": [InstanceGroup("train_group", "ml.g6.12xlarge", 2)]}
    )
    assert [args["instance_groups"][0].instance_type for args in groups] == [
        "ml.g6.12xlarge",
        "ml.g6e.12xlarge",
        "ml.g5.12xlarge",
        "ml.g4dn.12xlarge",
        "ml.p4d.24xlarge",
    ]
    assert all(args["instance_groups"][0].instance_count == 2 for args in groups)

    assert get_instance_type_fallbacks({"instance_type": "ml.p4de.24xlarge"}) == [
        {"instance_type": "ml.p4de.24xlarge"}
    ]
