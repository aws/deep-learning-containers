# MME model-management API coverage gaps

## What's covered end-to-end

- `POST /invocations` — every test file in this directory.
- `POST /models` (implicit via SM MME load path) — traversal guard exercised
  at SageMaker level in `test_mme_dynamic.py::test_mme_traversal_rejected`
  (parametrized with `../../evil.tar.gz` and `/etc/passwd`).

## What's NOT covered end-to-end

- `GET /models` (collection)
- `GET /models/{name}` (per-model status)
- `DELETE /models/{name}` (unload)
- LRU eviction under memory pressure
- Concurrent load of the same model

## Why not

These routes are handled by `python_service.PythonServiceResource`
(`scripts/docker/tensorflow/inference/sagemaker/python_service.py`, routes
at `on_get` / `on_delete`) but are only invoked from inside the container
by the SageMaker Model Manager sidecar. SageMaker's public `InvokeEndpoint`
API does not forward arbitrary HTTP paths — it only routes `/invocations`
plus MME `target_model` load semantics. Traversal payloads sent as
`target_model` therefore only exercise the guard on the POST path; the
GET / DELETE guards on the same code path have no external test surface.

## What a real coverage layer would look like

A container-level test harness that:

1. Pulls the built image from the PR's ECR tag.
2. `docker run` locally with the MME env
   (`SAGEMAKER_MULTI_MODEL=true`, port 8080 exposed).
3. Drives `POST /models`, `GET /models`, `GET /models/{name}`,
   `DELETE /models/{name}`, and `POST /invocations` via `requests`
   against `localhost:8080`.
4. Cleans up the container in a `finally`.

Blocked in this PR because:

- No existing docker-run-from-pytest pattern in this repo.
- Building one is substantial (~200 LOC harness + CI plumbing).
- Belongs in its own PR alongside a broader "container-level integration"
  story for all frameworks.

Prior attempt at unit-test coverage via Falcon TestClient
(`test_python_service_unit.py`) was reverted because `python_service.py`
monkey-patches SSL globally via gevent at module scope, poisoning
downstream boto3-using tests, and depends on the container-only
`/sagemaker/lock-file.lock` path.

## Bugs this gap allowed to ship past 24 green CI checks

All fixed in commits ff460f92 + d76611f4:

- `GET /models/{name}` returned 500 for every loaded model
  (`json.dumps(Response)` → `TypeError`, unhandled by `except ValueError`).
- MME traversal guard existed only on POST; GET/DELETE routes accepted the
  same path component unvalidated.
- `DELETE /models/{name}` leaked the dict entry on `ProcessLookupError`
  from `os.kill`, wedging the model name at 409 permanently.
