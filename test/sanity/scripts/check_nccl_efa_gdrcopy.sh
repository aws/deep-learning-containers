#!/usr/bin/env bash
set -uo pipefail

# NCCL / GDRCopy / EFA install checks for base devel images (no GPU required).
#
# Verifies the artifacts the Dockerfile installs are present, at the expected
# version, and dynamically loadable. Functional fabric coverage (EFA provider
# init, NCCL over Libfabric, GDRDMA) needs EFA hardware and lives in
# test/efa/test_efa.py, which runs on 2x p4d.24xlarge.
#
# Usage: check_nccl_efa_gdrcopy.sh <nccl_version> <efa_version> <gdrcopy_version>
# Example: check_nccl_efa_gdrcopy.sh 2.29.7-1 1.49.0 2.6

NCCL_VERSION="${1:?Usage: check_nccl_efa_gdrcopy.sh <nccl_version> <efa_version> <gdrcopy_version>}"
EFA_VERSION="${2:?Usage: check_nccl_efa_gdrcopy.sh <nccl_version> <efa_version> <gdrcopy_version>}"
GDRCOPY_VERSION="${3:?Usage: check_nccl_efa_gdrcopy.sh <nccl_version> <efa_version> <gdrcopy_version>}"
FAILED=0

pass() { echo "PASS: $1"; }
fail() {
  echo "FAIL: $1"
  FAILED=1
}

echo "=============== NCCL ==============="

# --- libnccl runtime is in the linker cache ---
if ldconfig -p 2>/dev/null | grep -q "libnccl\.so\.2"; then
  pass "libnccl.so.2 found in ldconfig"
else
  fail "libnccl.so.2 not found in ldconfig"
  ldconfig -p 2>/dev/null | grep -i nccl || echo "  (no libnccl entries at all)"
fi

# --- libnccl is loadable and reports the expected version ---
# Strip the RPM release suffix: "2.29.7-1" -> "2.29.7".
EXPECTED_NCCL="${NCCL_VERSION%%-*}"
# libnccl needs the NVIDIA driver's libcuda.so.1, which is injected by the
# container runtime and absent on CPU runners. Only assert the runtime version
# where the driver is actually present; the header check below covers the
# no-GPU case.
if ldconfig -p 2>/dev/null | grep -q "libcuda\.so\.1"; then
  NCCL_LOAD_ERROR=$(mktemp)
  NCCL_RUNTIME_VERSION=$(python3 -c "
import ctypes
lib = ctypes.CDLL('libnccl.so.2')
v = ctypes.c_int()
assert lib.ncclGetVersion(ctypes.byref(v)) == 0
# NCCL >= 2.9 encodes as major*10000 + minor*100 + patch. Base images ship far
# newer than that, so the pre-2.9 major*1000 encoding is not handled.
major, rest = divmod(v.value, 10000)
minor, patch = divmod(rest, 100)
print(f'{major}.{minor}.{patch}')
" 2>"$NCCL_LOAD_ERROR")
  if [ -z "$NCCL_RUNTIME_VERSION" ]; then
    fail "could not load libnccl.so.2 / call ncclGetVersion"
    cat "$NCCL_LOAD_ERROR"
  elif [ "$NCCL_RUNTIME_VERSION" = "$EXPECTED_NCCL" ]; then
    pass "ncclGetVersion reports $NCCL_RUNTIME_VERSION"
  else
    fail "NCCL version mismatch: runtime $NCCL_RUNTIME_VERSION, expected $EXPECTED_NCCL"
  fi
  rm -f "$NCCL_LOAD_ERROR"
else
  echo "INFO: libcuda.so.1 not present (no GPU runtime), skipping ncclGetVersion check"
fi

# --- NCCL development headers (needed to build NCCL-linked code in devel) ---
NCCL_HEADER=""
for CANDIDATE in /usr/include/nccl.h /usr/local/cuda/include/nccl.h /usr/local/include/nccl.h; do
  if [ -f "$CANDIDATE" ]; then
    NCCL_HEADER="$CANDIDATE"
    break
  fi
done
if [ -n "$NCCL_HEADER" ]; then
  pass "nccl.h found at $NCCL_HEADER"
  HEADER_VERSION=$(awk '
    /#define NCCL_MAJOR/ {major=$3}
    /#define NCCL_MINOR/ {minor=$3}
    /#define NCCL_PATCH/ {patch=$3}
    END {print major "." minor "." patch}
  ' "$NCCL_HEADER")
  if [ "$HEADER_VERSION" = "$EXPECTED_NCCL" ]; then
    pass "nccl.h reports $HEADER_VERSION"
  else
    fail "nccl.h version mismatch: $HEADER_VERSION, expected $EXPECTED_NCCL"
  fi
else
  fail "nccl.h not found (libnccl-devel missing?)"
fi

# --- The NCCL link-time symlink exists so -lnccl resolves ---
if ldconfig -p 2>/dev/null | grep -q "libnccl\.so " || \
  find /usr/lib64 /usr/lib /usr/local/lib /usr/local/cuda/lib64 \
    -maxdepth 1 -name 'libnccl.so' 2>/dev/null | grep -q .; then
  pass "libnccl.so link-time symlink present"
else
  fail "libnccl.so link-time symlink missing (-lnccl will not resolve)"
fi

# --- /etc/nccl.conf written by the EFA installer ---
if [ -f /etc/nccl.conf ]; then
  NCCL_CONF=$(cat /etc/nccl.conf)
  for SETTING in "NCCL_DEBUG=INFO" "NCCL_SOCKET_IFNAME"; do
    if echo "$NCCL_CONF" | grep -q "$SETTING"; then
      pass "/etc/nccl.conf contains $SETTING"
    else
      fail "/etc/nccl.conf missing $SETTING"
    fi
  done
else
  fail "/etc/nccl.conf not found"
fi

echo "=============== GDRCopy ==============="

# --- Userspace library. The gdrdrv kernel module comes from the host, so only
#     the library and its headers can be checked inside a container. ---
if [ -f /usr/local/lib/libgdrapi.so ]; then
  pass "/usr/local/lib/libgdrapi.so exists"
else
  fail "/usr/local/lib/libgdrapi.so not found"
  ls -la /usr/local/lib/libgdrapi* 2>/dev/null || echo "  (no libgdrapi* in /usr/local/lib)"
fi

# --- Versioned soname matches the requested GDRCopy release ---
if [ -e "/usr/local/lib/libgdrapi.so.${GDRCOPY_VERSION%%.*}" ]; then
  pass "libgdrapi soname libgdrapi.so.${GDRCOPY_VERSION%%.*} present"
else
  fail "libgdrapi.so.${GDRCOPY_VERSION%%.*} not found for GDRCopy ${GDRCOPY_VERSION}"
fi

# --- Library is in the linker cache and loadable, and exports the API ---
if ldconfig -p 2>/dev/null | grep -q libgdrapi; then
  pass "libgdrapi found in ldconfig"
else
  fail "libgdrapi not found in ldconfig"
fi

GDR_MISSING_DEPS=$(ldd /usr/local/lib/libgdrapi.so 2>&1 | grep "not found")
if [ -n "$GDR_MISSING_DEPS" ]; then
  fail "libgdrapi has unresolved shared libraries:"
  echo "$GDR_MISSING_DEPS"
else
  GDR_LOAD_ERROR=$(mktemp)
  if python3 -c "
import ctypes
lib = ctypes.CDLL('libgdrapi.so')
# gdr_open() would need the gdrdrv kernel module the host provides; only
# resolve the symbols to prove the API surface is there.
lib.gdr_open
lib.gdr_pin_buffer
" 2>"$GDR_LOAD_ERROR"; then
    pass "libgdrapi loadable and exports gdr_open/gdr_pin_buffer"
  else
    fail "libgdrapi not loadable or missing expected symbols"
    cat "$GDR_LOAD_ERROR"
  fi
  rm -f "$GDR_LOAD_ERROR"
fi

# --- Headers, so customers can build GDRCopy-linked code in the devel image ---
for HEADER in /usr/local/include/gdrapi.h /usr/local/include/gdrconfig.h; do
  if [ -f "$HEADER" ]; then
    pass "$HEADER exists"
  else
    fail "$HEADER not found"
  fi
done

echo "=============== EFA / libfabric ==============="

# --- EFA binaries on PATH ---
for BINARY in fi_info ibv_devinfo; do
  if command -v "$BINARY" &>/dev/null; then
    pass "$BINARY on PATH ($(command -v "$BINARY"))"
  else
    fail "$BINARY not found on PATH"
  fi
done

# --- libfabric reports a version. The exact libfabric release is vended by the
#     EFA installer, so log it rather than pinning it here. ---
if command -v fi_info &>/dev/null; then
  FI_VERSION_OUT=$(fi_info --version 2>&1)
  if echo "$FI_VERSION_OUT" | grep -qi libfabric; then
    pass "fi_info reports: $(echo "$FI_VERSION_OUT" | tr '\n' ' ')"
  else
    fail "fi_info --version did not report a libfabric version"
    echo "$FI_VERSION_OUT"
  fi

  # The efa provider must be compiled into libfabric. `fi_info -p efa` needs
  # real EFA hardware, but `fi_info -l` lists providers built into the library
  # and works on any host.
  if fi_info -l 2>/dev/null | grep -q efa; then
    pass "efa provider built into libfabric"
  else
    fail "efa provider not listed by fi_info -l"
    fi_info -l 2>&1 | head -20
  fi
fi

# --- aws-ofi-nccl plugin (the NCCL <-> libfabric bridge). EFA >= 1.44 installs
#     libnccl-net-ofi.so under lib64; older installers use the arch subdir. ---
OFI_PLUGIN=""
for CANDIDATE in /opt/amazon/ofi-nccl/lib64/libnccl-net-ofi.so \
  "/opt/amazon/ofi-nccl/lib/$(uname -m)-linux-gnu/libnccl-net.so"; do
  if [ -f "$CANDIDATE" ]; then
    OFI_PLUGIN="$CANDIDATE"
    break
  fi
done
if [ -n "$OFI_PLUGIN" ]; then
  pass "aws-ofi-nccl plugin found at $OFI_PLUGIN"
  # The plugin dlopen()s libcudart.so and links libfabric — an unresolved
  # dependency here is exactly the failure that silently drops NCCL to sockets.
  MISSING=$(ldd "$OFI_PLUGIN" 2>&1 | grep "not found")
  if [ -z "$MISSING" ]; then
    pass "aws-ofi-nccl plugin has no unresolved shared libraries"
  else
    fail "aws-ofi-nccl plugin has unresolved libraries:"
    echo "$MISSING"
  fi
else
  fail "aws-ofi-nccl plugin (libnccl-net*.so) not found under /opt/amazon/ofi-nccl"
  ls -laR /opt/amazon/ofi-nccl 2>/dev/null | head -20
fi

# --- EFA installer version. The installer records the packages it installed;
#     treat a missing file as informational since its path is not contractual. ---
if [ -f /opt/amazon/efa_installed_packages ]; then
  if grep -q "${EFA_VERSION}" /opt/amazon/efa_installed_packages; then
    pass "EFA installer ${EFA_VERSION} recorded in /opt/amazon/efa_installed_packages"
  else
    fail "EFA ${EFA_VERSION} not recorded in /opt/amazon/efa_installed_packages"
    cat /opt/amazon/efa_installed_packages
  fi
else
  echo "INFO: /opt/amazon/efa_installed_packages not present, skipping EFA version assertion"
fi

# --- rdma-core userspace, needed by the efa provider ---
if ldconfig -p 2>/dev/null | grep -q libibverbs; then
  pass "libibverbs found in ldconfig"
else
  fail "libibverbs not found in ldconfig (rdma-core missing?)"
fi

echo "=============== OpenMPI ==============="

# --- mpirun wrapper runs as root (the installer wraps it with
#     --allow-run-as-root; without that every multi-node launch fails).
#     Launch an absolute path: mpirun treats its first unrecognized token as the
#     executable, and AL2023 minimal ships no `hostname` binary, so a bare name
#     would fail on a missing binary rather than on OpenMPI. ---
if command -v mpirun &>/dev/null; then
  pass "mpirun on PATH ($(command -v mpirun))"
  if MPI_OUT=$(mpirun -n 1 /usr/bin/uname -n 2>&1); then
    pass "mpirun launched a rank as root"
  else
    fail "mpirun could not launch a rank as root (--allow-run-as-root wrapper broken?)"
    echo "$MPI_OUT"
  fi
else
  fail "mpirun not found on PATH"
fi

for SETTING in "hwloc_base_binding_policy = none" "rmaps_base_mapping_policy = slot"; do
  if grep -qF "$SETTING" /opt/amazon/openmpi/etc/openmpi-mca-params.conf 2>/dev/null; then
    pass "openmpi-mca-params.conf contains '$SETTING'"
  else
    fail "openmpi-mca-params.conf missing '$SETTING'"
  fi
done

echo "=============== SSH (multi-node launch) ==============="

# --- sshd + keys, used by mpirun to reach worker nodes ---
if [ -x /usr/sbin/sshd ]; then
  pass "/usr/sbin/sshd is executable"
else
  fail "/usr/sbin/sshd not found or not executable"
fi

if [ -f /root/.ssh/authorized_keys ]; then
  pass "/root/.ssh/authorized_keys exists"
else
  fail "/root/.ssh/authorized_keys not found"
fi

if grep -q "StrictHostKeyChecking no" /root/.ssh/config 2>/dev/null; then
  pass "StrictHostKeyChecking disabled in /root/.ssh/config"
else
  fail "StrictHostKeyChecking not disabled in /root/.ssh/config"
fi

echo "=============== Environment ==============="

# --- PATH / LD_LIBRARY_PATH must expose the EFA + OpenMPI trees, otherwise the
#     libraries are installed but unreachable at runtime. ---
for DIRECTORY in /opt/amazon/openmpi/bin /opt/amazon/efa/bin /usr/local/cuda/bin; do
  case ":${PATH}:" in
    *":${DIRECTORY}:"*) pass "PATH contains ${DIRECTORY}" ;;
    *) fail "PATH missing ${DIRECTORY}" ;;
  esac
done

for DIRECTORY in /opt/amazon/ofi-nccl/lib64 /opt/amazon/openmpi/lib /opt/amazon/efa/lib /usr/local/lib; do
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${DIRECTORY}:"*) pass "LD_LIBRARY_PATH contains ${DIRECTORY}" ;;
    *) fail "LD_LIBRARY_PATH missing ${DIRECTORY}" ;;
  esac
done

exit $FAILED
