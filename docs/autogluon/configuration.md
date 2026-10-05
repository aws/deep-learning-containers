# Configuration

## Inference Handler

At startup the server imports the handler file named by `SAGEMAKER_PROGRAM` from `/opt/ml/model/code/` (the `code/` directory of your model artifact).
The handler must define two functions:

| Function | Signature | Purpose |
| --- | --- | --- |
| `model_fn` | `model_fn(model_dir) -> model` | Called once per worker at startup with `/opt/ml/model`. Load and return your predictor. |
| `transform_fn` | `transform_fn(model, request_body, input_content_type, output_content_type) -> (body, content_type)` | Called for every `/invocations` request. |

Request and response handling:

- `request_body` is a `str` for `text/*`, `application/json`, and `application/jsonl` requests, and `bytes` for every other content type.
- `input_content_type` comes from the `Content-Type` header (default `application/json`). `output_content_type` comes from the `Accept` header,
  falling back to `SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT`.
- `transform_fn` must return a `(body, content_type)` tuple where `body` is `str` or `bytes`.
- Raising `ValueError` returns **HTTP 400** with the error message. Any other exception returns **HTTP 500** and is logged to CloudWatch.
- `input_fn`, `predict_fn`, and `output_fn` are not used. Put that logic in `transform_fn`.

## Extra Python Packages

If the model artifact contains `code/requirements.txt`, the container installs it with `uv pip install` once at startup, before the server starts. To
install from a private AWS CodeArtifact repository, set `CA_REPOSITORY_ARN` (see below); the execution role needs
`codeartifact:GetAuthorizationToken`, `codeartifact:GetRepositoryEndpoint`, `codeartifact:ReadFromRepository`, and `sts:GetServiceBearerToken`.

## Environment Variables

These variables apply to the `serve` mode. Set them in the {{ sm_short }} model's `Environment`.

| Variable | Default | Description |
| --- | --- | --- |
| `SAGEMAKER_PROGRAM` | `inference.py` | Handler file, relative to `/opt/ml/model/code/` (or an absolute path) |
| `SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT` | `application/json` | `output_content_type` passed to `transform_fn` when the request has no `Accept` header |
| `SAGEMAKER_MODEL_SERVER_WORKERS` | `1` | Number of Gunicorn worker processes. Each worker loads its own copy of the model. |
| `SAGEMAKER_MODEL_SERVER_TIMEOUT` | `60` | Per-request worker timeout, in seconds |
| `SAGEMAKER_CONTAINER_LOG_LEVEL` | `INFO` | Server log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, or the numeric equivalent) |
| `CA_REPOSITORY_ARN` | unset | CodeArtifact repository ARN (`arn:aws:codeartifact:<region>:<account>:repository/<domain>/<repo>`) used as the package index for `code/requirements.txt` |

{{ sm_short }} sets `SAGEMAKER_BIND_TO_PORT` automatically; the server listens on port 8080 otherwise.

## Known Limitations

- **x86 only.** There are no ARM64 images.
- **One model per container.** Multi-model endpoints are not supported.
- **Model loading happens before `/ping` succeeds.** Raise `ContainerStartupHealthCheckTimeoutInSeconds` for large predictors.
