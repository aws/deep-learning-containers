"""SageMaker inference handler for OpenFold3.

Implements the SageMaker contract: model_fn, input_fn, predict_fn, output_fn.

Two predict paths:
  - subprocess (default): shell out to `run_openfold predict` per request.
    Stable, isolated, but reloads the ~2GB model into GPU memory every time.
  - in_memory: keep the lightning module resident across requests. Build a
    fresh data module + trainer per request, call trainer.predict() directly.
    ~3-4x faster warm latency for small proteins. Selected via
    options.engine="in_memory" or env var OPENFOLD3_ENGINE=in_memory.

Other behaviors:
  - CIF/JSON outputs are returned in full, never truncated.
  - MSA is OFF by default (SageMaker network isolation has no outbound
    internet). Callers opt in via options.use_msa_server=true.
  - Per-GPU warmup at process startup compiles CUDA kernels before /ping
    returns 200, hiding the ~6-min one-time compile from the customer.

Tunables (env vars):
  OPENFOLD3_ENGINE                "subprocess" (default) or "in_memory"
  OPENFOLD3_NUM_GPUS              override detected GPU count
  OPENFOLD3_WARMUP_SIZES          residue counts for warmup (default: "30")
  OPENFOLD3_SKIP_WARMUP           "1" to skip warmup entirely
  OPENFOLD3_PREDICT_TIMEOUT_SEC   subprocess timeout
"""

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_PREDICT_TIMEOUT_SEC = int(os.environ.get("OPENFOLD3_PREDICT_TIMEOUT_SEC", "3300"))
_DEFAULT_ENGINE = os.environ.get("OPENFOLD3_ENGINE", "subprocess").lower()


def _detected_gpu_count() -> int:
    override = os.environ.get("OPENFOLD3_NUM_GPUS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning("Invalid OPENFOLD3_NUM_GPUS=%r, falling back to detected", override)
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


class _GpuPool:
    def __init__(self, num_gpus: int):
        self._q: "queue.Queue[int]" = queue.Queue()
        for i in range(num_gpus):
            self._q.put(i)
        self.size = num_gpus

    def acquire(self, timeout: Optional[float] = None) -> int:
        return self._q.get(timeout=timeout)

    def release(self, gpu_index: int) -> None:
        self._q.put(gpu_index)


class _GpuLease:
    def __init__(self, pool: _GpuPool):
        self._pool = pool
        self.gpu_index: Optional[int] = None

    def __enter__(self) -> int:
        self.gpu_index = self._pool.acquire()
        return self.gpu_index

    def __exit__(self, *exc) -> None:
        if self.gpu_index is not None:
            self._pool.release(self.gpu_index)
            self.gpu_index = None


class OpenFold3Predictor:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        num_gpus = _detected_gpu_count()
        if num_gpus < 1:
            logger.warning("No GPUs detected; falling back to CPU pool of size 1")
            num_gpus = 1
        self.num_gpus = num_gpus
        self._gpu_pool = _GpuPool(num_gpus)
        # Lazily initialized in-memory state. Built on first in_memory request,
        # then reused. Subprocess path doesn't touch any of this.
        self._inmem_lock = threading.Lock()
        self._inmem_runner = None
        self._inmem_lightning_module = None
        logger.info(
            "Predictor init: num_gpus=%d engine_default=%s checkpoint=%s",
            num_gpus,
            _DEFAULT_ENGINE,
            self.checkpoint_path,
        )

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._predict_on_gpu(input_data, gpu_index=None)

    def _predict_on_gpu(
        self, input_data: Dict[str, Any], gpu_index: Optional[int]
    ) -> Dict[str, Any]:
        options = input_data.get("options", {}) or {}

        if "inputs" in input_data:
            openfold_data = self._convert_to_openfold_format(input_data)
        elif "queries" in input_data:
            openfold_data = input_data
        else:
            return {"status": "error", "error": "Input must contain 'inputs' or 'queries'"}

        engine = (options.get("engine") or _DEFAULT_ENGINE).lower()
        if engine == "in_memory":
            return self._predict_in_memory(openfold_data, options)

        # Default: subprocess path with GPU pinning.
        all_gpus = bool(options.get("all_gpus", False))
        lease: Optional[_GpuLease] = None
        if all_gpus:
            held = [self._gpu_pool.acquire() for _ in range(self.num_gpus)]
            cuda_visible = ",".join(str(i) for i in held)
            try:
                return self._run_predict_subprocess(
                    openfold_data,
                    options,
                    cuda_visible=cuda_visible,
                    gpu_label=f"all={cuda_visible}",
                )
            finally:
                for g in held:
                    self._gpu_pool.release(g)
        else:
            if gpu_index is None:
                lease = _GpuLease(self._gpu_pool)
                gpu_index = lease.__enter__()
            try:
                return self._run_predict_subprocess(
                    openfold_data,
                    options,
                    cuda_visible=str(gpu_index),
                    gpu_label=f"gpu={gpu_index}",
                )
            finally:
                if lease is not None:
                    lease.__exit__(None, None, None)

    def _run_predict_subprocess(
        self,
        openfold_data: Dict[str, Any],
        options: Dict[str, Any],
        cuda_visible: str,
        gpu_label: str,
    ) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            query_path = os.path.join(temp_dir, "query.json")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)

            with open(query_path, "w") as f:
                json.dump(openfold_data, f)

            cmd = [
                "run_openfold",
                "predict",
                f"--query_json={query_path}",
                f"--output_dir={output_dir}",
                f"--inference_ckpt_path={self.checkpoint_path}",
            ]
            cmd.extend(["--use_msa_server", "true" if options.get("use_msa_server") else "false"])
            cmd.extend(["--use_templates", "true" if options.get("use_templates") else "false"])

            if "num_model_seeds" in options:
                cmd.append(f"--num_model_seeds={int(options['num_model_seeds'])}")
            if "num_diffusion_samples" in options:
                cmd.append(f"--num_diffusion_samples={int(options['num_diffusion_samples'])}")

            runner_yaml_path = options.get("runner_yaml")
            inline_yaml = options.get("runner_yaml_inline")
            if inline_yaml and not runner_yaml_path:
                runner_yaml_path = os.path.join(temp_dir, "runner.yml")
                with open(runner_yaml_path, "w") as f:
                    f.write(inline_yaml)
            if runner_yaml_path:
                cmd.append(f"--runner_yaml={runner_yaml_path}")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible

            logger.info("[%s engine=subprocess] Running: %s", gpu_label, " ".join(cmd))
            t0 = time.time()
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=_PREDICT_TIMEOUT_SEC,
                env=env,
            )
            dt = time.time() - t0

            if result.returncode != 0:
                tail = result.stdout[-4000:] if result.stdout else ""
                logger.error(
                    "[%s engine=subprocess] Failed (rc=%s, %.1fs): %s",
                    gpu_label,
                    result.returncode,
                    dt,
                    tail,
                )
                return {
                    "status": "error",
                    "error": f"Prediction failed with exit code {result.returncode}",
                    "log_tail": tail,
                }

            logger.info("[%s engine=subprocess] Completed in %.1fs", gpu_label, dt)
            return self._parse_output(output_dir)

    # ------------------------------------------------------------------
    # In-memory predict path.
    # ------------------------------------------------------------------
    def _ensure_inmem_initialized(self) -> None:
        """Build the lightning module + experiment runner once. Idempotent."""
        if self._inmem_runner is not None:
            return
        with self._inmem_lock:
            if self._inmem_runner is not None:
                return
            logger.info("[engine=in_memory] Initializing resident model...")
            t0 = time.time()
            from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
            from openfold3.entry_points.validator import InferenceExperimentConfig

            expt_config = InferenceExperimentConfig(
                inference_ckpt_path=Path(self.checkpoint_path),
            )
            runner = InferenceExperimentRunner(
                expt_config,
                num_diffusion_samples=None,
                num_model_seeds=None,
                use_msa_server=False,
                use_templates=False,
                output_dir=None,
            )
            runner.setup()  # loads weights into GPU memory
            _ = runner.lightning_module  # trigger instantiation, populate cache

            self._inmem_runner = runner
            self._inmem_lightning_module = runner.lightning_module
            logger.info("[engine=in_memory] Model resident in %.1fs", time.time() - t0)

    def _predict_in_memory(
        self,
        openfold_data: Dict[str, Any],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a prediction without spawning a subprocess.

        Builds a fresh data module + trainer per request, uses the resident
        lightning module. Holds a process-wide lock for the duration because
        the resident module + trainer aren't safe to call concurrently.
        For multi-tenant concurrency, use the subprocess path instead.
        """
        try:
            self._ensure_inmem_initialized()
        except Exception:
            logger.exception("[engine=in_memory] Initialization failed; falling back to subprocess")
            # Fall back so a transient init failure doesn't permanently break the endpoint.
            return self._run_predict_subprocess(
                openfold_data,
                options,
                cuda_visible="0",
                gpu_label="gpu=0(fallback)",
            )

        # Lazy imports — only paid when in_memory engine is used.
        import pytorch_lightning as pl
        from openfold3.core.data.framework.data_module import DataModuleConfig
        from openfold3.core.runners.writer import OF3OutputWriter
        from openfold3.projects.of3_all_atom.config.dataset_configs import (
            InferenceDatasetSpec,
            InferenceJobConfig,
        )
        from openfold3.projects.of3_all_atom.config.inference_query_format import (
            InferenceQuerySet,
        )

        # Serialize in-memory predictions: trainer + lightning module are
        # shared resources. Acquire all GPUs from the pool to block subprocess
        # path requests from interfering.
        held = [self._gpu_pool.acquire() for _ in range(self.num_gpus)]
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "output"
                output_dir.mkdir()

                # Build query set in memory by writing query JSON to temp file
                # (InferenceQuerySet.from_json reads from disk).
                query_path = Path(temp_dir) / "query.json"
                query_path.write_text(json.dumps(openfold_data))
                query_set = InferenceQuerySet.from_json(query_path)

                runner = self._inmem_runner

                # Apply optional per-request overrides without mutating runner state.
                num_seeds = options.get("num_model_seeds")
                if num_seeds:
                    from openfold3.utils.seed_utils import generate_seeds

                    seeds = generate_seeds(42, int(num_seeds))
                else:
                    seeds = runner.seeds

                # Build a fresh data module bound to THIS query.
                inference_config = InferenceJobConfig(
                    query_set=query_set,
                    seeds=seeds,
                    ccd_file_path=runner.dataset_config_kwargs.ccd_file_path,
                    msa=runner.dataset_config_kwargs.msa,
                    template=runner.dataset_config_kwargs.template,
                    template_preprocessor_settings=runner.experiment_config.template_preprocessor_settings,
                )
                inference_spec = InferenceDatasetSpec(config=inference_config)
                data_module_config = DataModuleConfig(
                    datasets=[inference_spec],
                    **runner.data_module_args.model_dump(),
                )
                from openfold3.core.data.framework.data_module import InferenceDataModule

                data_module = InferenceDataModule(
                    data_module_config,
                    use_msa_server=bool(options.get("use_msa_server", False)),
                    use_templates=bool(options.get("use_templates", False)),
                    msa_computation_settings=runner.experiment_config.msa_computation_settings,
                )

                # Build a fresh writer pointed at THIS request's output dir.
                writer = OF3OutputWriter(
                    output_dir=output_dir,
                    **runner.output_writer_settings.model_dump(),
                )

                # Build a fresh trainer with single-GPU inference settings.
                # We pin to one GPU explicitly to avoid DDP overhead.
                trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=1,
                    num_nodes=1,
                    logger=False,
                    callbacks=[writer],
                    enable_progress_bar=False,
                    enable_model_summary=False,
                )

                logger.info("[engine=in_memory] Predicting (%d seeds)", len(seeds))
                t0 = time.time()
                trainer.predict(
                    model=self._inmem_lightning_module,
                    datamodule=data_module,
                    return_predictions=False,
                )
                dt = time.time() - t0
                logger.info("[engine=in_memory] Completed in %.1fs", dt)

                return self._parse_output(str(output_dir))
        finally:
            for g in held:
                self._gpu_pool.release(g)

    # ------------------------------------------------------------------
    # Shared utilities.
    # ------------------------------------------------------------------
    def _convert_to_openfold_format(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        input_spec = input_data["inputs"][0]
        molecules = input_spec.get("molecules", [])
        query_id = input_spec.get("input_id", "prediction")

        chains = []
        for mol in molecules:
            mol_type = mol.get("type", "protein")
            raw_id = mol.get("id")
            if isinstance(raw_id, list):
                chain_ids = raw_id
            elif isinstance(raw_id, str):
                chain_ids = [raw_id]
            else:
                chain_ids = ["A"]

            chain: Dict[str, Any] = {"molecule_type": mol_type, "chain_ids": chain_ids}
            if mol_type in ("protein", "dna", "rna"):
                chain["sequence"] = mol.get("sequence", "")
            elif mol_type == "ligand":
                if "ccd_codes" in mol:
                    chain["ccd_codes"] = mol["ccd_codes"]
                elif "smiles" in mol:
                    chain["smiles"] = mol["smiles"]
            chains.append(chain)

        return {"queries": {query_id: {"chains": chains}}}

    def _parse_output(self, output_dir: str) -> Dict[str, Any]:
        output_path = Path(output_dir)
        results: Dict[str, Any] = {"status": "success", "structures": [], "confidences": []}

        for cif_file in sorted(output_path.rglob("*.cif")):
            try:
                content = cif_file.read_text()
            except Exception as e:
                logger.warning("Could not read %s: %s", cif_file, e)
                continue
            results["structures"].append(
                {
                    "filename": cif_file.name,
                    "relative_path": str(cif_file.relative_to(output_path)),
                    "format": "cif",
                    "content": content,
                    "size_bytes": len(content),
                }
            )

        for json_file in sorted(output_path.rglob("*confidences*.json")):
            try:
                results["confidences"].append(
                    {
                        "filename": json_file.name,
                        "relative_path": str(json_file.relative_to(output_path)),
                        "data": json.loads(json_file.read_text()),
                    }
                )
            except Exception as e:
                logger.warning("Could not parse %s: %s", json_file, e)

        if not results["structures"]:
            logger.warning("No CIF files found in %s", output_path)
            results["status"] = "error"
            results["error"] = "OpenFold3 produced no CIF output"

        return results

    def warmup(self) -> None:
        """Compile CUDA kernels on every GPU in the pool, in parallel.

        Always uses the subprocess path so the resident in-memory state isn't
        bound to a specific GPU at warmup time (in-memory init happens lazily
        on first in_memory request).
        """
        if not torch.cuda.is_available():
            logger.info("Skipping warmup: no CUDA device")
            return

        sizes_env = os.environ.get("OPENFOLD3_WARMUP_SIZES", "30")
        try:
            sizes = [int(s) for s in sizes_env.split(",") if s.strip()]
        except ValueError:
            logger.warning("Invalid OPENFOLD3_WARMUP_SIZES=%r, falling back to [30]", sizes_env)
            sizes = [30]

        UNIT = "MQIFVKT"
        warmup_queries = []
        for n in sizes:
            seq = (UNIT * ((n // len(UNIT)) + 1))[:n]
            warmup_queries.append(
                {
                    "queries": {
                        f"warmup_{n}": {
                            "chains": [
                                {
                                    "molecule_type": "protein",
                                    "chain_ids": ["A"],
                                    "sequence": seq,
                                }
                            ]
                        }
                    },
                    "options": {
                        "use_msa_server": False,
                        "use_templates": False,
                        "engine": "subprocess",
                    },
                }
            )

        def warm_one_gpu(gpu_idx: int) -> None:
            for q in warmup_queries:
                seq_len = len(next(iter(q["queries"].values()))["chains"][0]["sequence"])
                logger.info("[gpu=%d] Warmup: %d-residue prediction", gpu_idx, seq_len)
                t0 = time.time()
                try:
                    result = self._predict_on_gpu(q, gpu_index=gpu_idx)
                    dt = time.time() - t0
                    if result.get("status") == "success":
                        logger.info(
                            "[gpu=%d] Warmup %d-residue: succeeded in %.1fs", gpu_idx, seq_len, dt
                        )
                    else:
                        logger.warning(
                            "[gpu=%d] Warmup %d-residue: error after %.1fs: %s",
                            gpu_idx,
                            seq_len,
                            dt,
                            result.get("error"),
                        )
                except Exception:
                    logger.exception("[gpu=%d] Warmup %d-residue: failed", gpu_idx, seq_len)

        t_total = time.time()
        with ThreadPoolExecutor(max_workers=self.num_gpus) as ex:
            futs = [ex.submit(warm_one_gpu, i) for i in range(self.num_gpus)]
            for f in as_completed(futs):
                f.result()
        logger.info(
            "Warmup complete across %d GPU(s) in %.1fs", self.num_gpus, time.time() - t_total
        )


def _find_checkpoint() -> str:
    openfold_cache = os.environ.get("OPENFOLD_CACHE", os.path.expanduser("~/.openfold3"))
    preferred = ["of3-p2-155k.pt", "of3-p2-145k.pt"]

    if os.path.exists(openfold_cache):
        for name in preferred:
            path = os.path.join(openfold_cache, name)
            if os.path.exists(path):
                return path
        for f in os.listdir(openfold_cache):
            if f.endswith((".pt", ".ckpt")):
                return os.path.join(openfold_cache, f)

    raise FileNotFoundError(f"No checkpoint found in {openfold_cache}")


def model_fn(model_dir: str):
    logger.info("Loading model (model_dir=%s)", model_dir)
    checkpoint_path = _find_checkpoint()
    predictor = OpenFold3Predictor(checkpoint_path)
    if os.environ.get("OPENFOLD3_SKIP_WARMUP", "").lower() not in ("1", "true", "yes"):
        predictor.warmup()
    return predictor


def input_fn(request_body: bytes, content_type: str = "application/json"):
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: Dict[str, Any], model: OpenFold3Predictor):
    return model.predict(input_data)


def output_fn(prediction: Dict[str, Any], accept: str = "application/json"):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
