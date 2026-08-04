# MME model-management API coverage gaps

## What's covered end-to-end

- `POST /invocations` — every test file in this directory.

## What's NOT covered end-to-end

- `POST /models` traversal guard — code-reviewed only, not end-to-end
  tested. SageMaker Runtime's `target_model` header maps to an S3 key
  lookup on the MME prefix, not to the container's `POST /models`
  route directly, so a traversal payload sent as `target_model` never
  reaches the container-side guard in
  `python_service._handle_load_model_post`. Botocore does not enforce
  a `pattern` on the `TargetModel` param either (only min/max length),
  so a client-side reject is not guaranteed. A meaningful end-to-end
  test needs a direct-to-container harness (see below).
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
1. `docker run` locally with the MME env
   (`SAGEMAKER_MULTI_MODEL=true`, port 8080 exposed).
1. Drives `POST /models`, `GET /models`, `GET /models/{name}`,
   `DELETE /models/{name}`, and `POST /invocations` via `requests`
   against `localhost:8080`.
1. Cleans up the container in a `finally`.

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

## TFS 2.20 x TF 2.21 forward-compat boundary — `DecodeJxl`

TF 2.21 adds `tf.raw_ops.DecodeJxl` (JPEG XL image decoder). TFS 2.20's
kernel registry does not know this op. A customer SavedModel that
invokes `DecodeJxl` will:

- **`Model.create`** — succeeds (SavedModel proto validation only checks
  op name / attr shape, not that a kernel is registered).
- **`Endpoint.create`** — reports `InService`; TFS boots and reports the
  model as `AVAILABLE` on `/models/{name}`.
- **First `predict`** — expected to return 4xx with `"Op type not registered 'DecodeJxl'"`.

**Coverage status**: an integration test attempting to pin this boundary
was added and then removed. In practice, when the test host has
TF 2.21 installed, the exported model reached the endpoint in a form
that TFS 2.20 accepted (op likely inlined or optimized away during
export), so the test could not reliably distinguish "TFS rejected the
op" from "SM/TFS returned an empty error body for another reason." A
proper boundary test needs either a direct-to-TFS harness (bypassing
SageMaker's response reshaping) or a pre-built SavedModel artifact
committed to the repo that guarantees the op survives export.

**Customer guidance until this is properly covered**: if a customer's
model uses JPEG XL decode, they must decode at the client and send
raw pixels to the endpoint, or wait for a TFS 2.21+ image.
