"""Record which MXFP4 MoE kernels this image can actually reach.

With moe_runner_backend='auto' sglang picks trtllm-gen on SM100/SM103 when its cubins are
present and silently drops to marlin when they are not, so an image can ship on a slower
expert path unnoticed. MXFP4 is Kimi-K3's routed-expert format, which is what makes this
worth recording at build time.

Deliberately non-fatal: trtllm-gen's MoE cubins are unmerged upstream and a version bump
does not help, so a missing trtllm-gen is the expected result today rather than a defect.
Probing trtllm_fp4_block_scale_moe mirrors what sglang itself gates on. A positive is
necessary but not sufficient -- the symbol resolving does not prove cubins exist for a
given problem shape.
"""

import importlib

PROBES = (
    ("flashinfer_mxfp4/trtllm-gen", "flashinfer", "trtllm_fp4_block_scale_moe"),
    ("deep_gemm", "deep_gemm", None),
    ("marlin", "sglang.srt.layers.quantization.marlin_utils_fp4", None),
)

backends = []
for name, module, attr in PROBES:
    try:
        mod = importlib.import_module(module)
        if attr is not None and not hasattr(mod, attr):
            raise AttributeError(f"{module}.{attr} missing")
        backends.append(name)
    except Exception as exc:  # noqa: BLE001 - any failure means the backend is unreachable
        print(f"MXFP4 MoE: {name} NOT available ({type(exc).__name__}: {exc})")

print("MXFP4_MOE_BACKENDS=" + (",".join(backends) if backends else "none"))
if "flashinfer_mxfp4/trtllm-gen" not in backends:
    print("NOTE: moe_runner_backend='auto' will fall back; expected until cubins land upstream.")
