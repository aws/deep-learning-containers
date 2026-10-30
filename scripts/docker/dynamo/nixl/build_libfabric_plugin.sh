#!/bin/bash
# Build NIXL's LIBFABRIC (EFA) plugin for the NIXL the backend ships, when it needs one:
#   - a nixl-cuXX wheel (vLLM, SGLang) whose version has patches in <patch-dir>/<nixl-version>/
#     (upstream fixes not in a NIXL release yet), or
#   - the NIXL bundled in TensorRT-LLM (tensorrt_llm/libs/nixl), which ships without LIBFABRIC; its
#     version isn't recorded in the package, so pass it as BUNDLED_NIXL_VERSION.
# Otherwise, and in a base without EFA, the shipped plugins are left as they are.
#
#   build_libfabric_plugin.sh build <patch-dir> <out-dir>   # build stage, same base and environment
#   build_libfabric_plugin.sh install <out-dir>             # final stage
set -euo pipefail

NIXL_REPO=${NIXL_REPO:-https://github.com/ai-dynamo/nixl.git}
PY=$(command -v python3)

# Where the backend's NIXL lives: its name, version, plugin directory, and an existing plugin to copy
# the linking (RPATH, renamed libnuma) from.
read -r NIXL_DIST NIXL_VERSION PLUGIN_DIR REF_PLUGIN < <(BUNDLED_NIXL_VERSION=${BUNDLED_NIXL_VERSION:-} "$PY" - <<'EOF'
import importlib.metadata as m
import importlib.util
import os
for name in ("nixl-cu13", "nixl-cu12"):
    try:
        d = m.distribution(name)
    except m.PackageNotFoundError:
        continue
    name, version = name, d.version
    plugin_dir = str(d.locate_file(f".{name.replace('-', '_')}.mesonpy.libs/plugins"))
    break
else:
    spec = importlib.util.find_spec("tensorrt_llm")  # locate only; importing needs a GPU driver
    if spec is None:
        raise SystemExit("no nixl-cu12/nixl-cu13 wheel and no tensorrt_llm installed")
    version = os.environ.get("BUNDLED_NIXL_VERSION")
    if not version:
        raise SystemExit("tensorrt_llm bundles NIXL: set BUNDLED_NIXL_VERSION")
    name = "tensorrt_llm-bundled-nixl"
    plugin_dir = os.path.join(spec.submodule_search_locations[0], "libs", "nixl", "plugins")
ref = next(p for p in (os.path.join(plugin_dir, f"libplugin_{b}.so") for b in ("LIBFABRIC", "UCX"))
           if os.path.exists(p))
print(name, version, plugin_dir, ref)
EOF
)
PLUGIN=${PLUGIN_DIR}/libplugin_LIBFABRIC.so

case "${1:-}" in
build)
    PATCH_DIR=$2/${NIXL_VERSION} OUT=$3
    mkdir -p "$OUT"
    if [ ! -f /opt/amazon/efa/include/rdma/fabric.h ]; then
        echo "no EFA/libfabric in this base; keeping the shipped NIXL plugins"
        exit 0
    fi
    PATCHES=$(ls "$PATCH_DIR"/*.patch 2>/dev/null || true)
    if [ -z "$PATCHES" ] && [ -f "$PLUGIN" ]; then
        echo "${NIXL_DIST} ${NIXL_VERSION}: no patches in ${PATCH_DIR}; keeping the shipped plugin"
        exit 0
    fi
    CUDA_DASH=$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr . -)
    PYVER=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    dnf install -y --setopt=install_weak_deps=False \
        gcc gcc-c++ git hwloc-devel numactl-devel python${PYVER}-devel \
        cuda-cudart-devel-${CUDA_DASH} cuda-driver-devel-${CUDA_DASH}
    python${PYVER} -m venv /tmp/nixl-tools
    /tmp/nixl-tools/bin/pip install --no-cache-dir meson ninja patchelf "pybind11[global]"
    export PATH=/tmp/nixl-tools/bin:${PATH}
    # Link against the driver stub; the node's driver provides libcuda.so.1 at runtime.
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}

    git clone -q --depth 1 --branch "v${NIXL_VERSION}" "$NIXL_REPO" /tmp/nixl
    cd /tmp/nixl
    for p in $PATCHES; do echo "applying $(basename "$p")"; git apply "$p"; done
    # Python bindings are configured but not built; only the plugin target is.
    meson setup build -Dbuildtype=release -Denable_plugins=LIBFABRIC -Dlibfabric_path=/opt/amazon/efa \
        -Dbuild_tests=false -Dbuild_examples=false -Dinstall_headers=false \
        -Ddisable_gds_backend=true -Ddisable_mooncake_backend=true -Ddisable_infinia_backend=true
    ninja -C build src/plugins/libfabric/libplugin_LIBFABRIC.so
    NEW=build/src/plugins/libfabric/libplugin_LIBFABRIC.so

    # Link it like the shipped plugins: same RPATH and the same (possibly auditwheel-renamed) libnuma,
    # so it resolves libnixl_common/libserdes/libnuma from the backend's NIXL, not this build tree.
    patchelf --set-rpath "$(patchelf --print-rpath "$REF_PLUGIN")" "$NEW"
    NUMA=$(patchelf --print-needed "$REF_PLUGIN" | grep '^libnuma' || true)
    if [ -n "$NUMA" ] && [ "$NUMA" != libnuma.so.1 ]; then patchelf --replace-needed libnuma.so.1 "$NUMA" "$NEW"; fi
    cp "$NEW" "$OUT/libplugin_LIBFABRIC.so"
    for p in $PATCHES; do basename "$p"; done > "$OUT/PATCHES"
    echo "${NIXL_DIST} ${NIXL_VERSION}" > "$OUT/VERSION"
    ;;
install)
    OUT=$2
    if [ ! -f "$OUT/libplugin_LIBFABRIC.so" ]; then
        echo "${NIXL_DIST} ${NIXL_VERSION}: no rebuilt LIBFABRIC plugin; keeping the shipped plugins"
        exit 0
    fi
    [ "$(cat "$OUT/VERSION")" = "${NIXL_DIST} ${NIXL_VERSION}" ] \
        || { echo "plugin built for $(cat "$OUT/VERSION"), installed ${NIXL_DIST} ${NIXL_VERSION}" >&2; exit 1; }
    install -m 0755 "$OUT/libplugin_LIBFABRIC.so" "$PLUGIN"
    # libcuda.so.1 comes from the driver at runtime; use the CUDA compat copy to check the rest resolves.
    export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    if ldd "$PLUGIN" | grep 'not found'; then
        exit 1
    fi
    if [ "$NIXL_DIST" = tensorrt_llm-bundled-nixl ]; then
        # No NIXL Python API here; resolving every symbol against the bundled NIXL is the check.
        "$PY" -c "import ctypes, os; ctypes.CDLL('$PLUGIN', mode=os.RTLD_NOW)"
    else
        "$PY" -c "
from nixl._api import nixl_agent, nixl_agent_config
plugins = nixl_agent('nixl-plugin-check', nixl_agent_config(backends=[])).get_plugin_list()
assert 'LIBFABRIC' in plugins, plugins"
    fi
    mkdir -p /opt/nixl
    cp "$OUT/VERSION" /opt/nixl/LIBFABRIC_PLUGIN_SOURCE
    if [ -s "$OUT/PATCHES" ]; then cp "$OUT/PATCHES" /opt/nixl/LIBFABRIC_PLUGIN_PATCHES; fi
    echo "${NIXL_DIST} ${NIXL_VERSION}: LIBFABRIC plugin built, patches: $(tr '\n' ' ' < "$OUT/PATCHES")"
    ;;
*)
    echo "usage: $0 build <patch-dir> <out-dir> | install <out-dir>" >&2
    exit 2
    ;;
esac
