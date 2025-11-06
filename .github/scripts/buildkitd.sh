#!/usr/bin/env bash
set -euo pipefail

# Detect architecture (supports x86_64 and aarch64)
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)   ARCH="amd64" ;;
  aarch64)  ARCH="arm64" ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

# Check if buildkitd is already installed
if command -v buildkitd >/dev/null 2>&1; then
  echo "✅ BuildKit already installed: $(buildkitd --version)"
  exit 0
fi

echo "🔍 Fetching latest BuildKit release..."

# Get latest release tag from GitHub API
LATEST_TAG=$(curl -s https://api.github.com/repos/moby/buildkit/releases/latest | \
             grep -Po '"tag_name":\s*"\K[^"]+')

if [ -z "$LATEST_TAG" ]; then
  echo "❌ Failed to fetch latest release tag."
  exit 1
fi

echo "📦 Latest BuildKit release: $LATEST_TAG"

# Download and install
TMPDIR=$(mktemp -d)
URL="https://github.com/moby/buildkit/releases/download/${LATEST_TAG}/buildkit-${LATEST_TAG}.linux-${ARCH}.tar.gz"
echo "⬇️  Downloading $URL"
curl -L "$URL" -o "$TMPDIR/buildkit.tar.gz"

echo "📂 Extracting and installing..."
sudo tar -C /usr/local -xzf "$TMPDIR/buildkit.tar.gz"

# Optional: ensure it's on PATH
if ! command -v buildkitd >/dev/null; then
  echo "⚠️ buildkitd not in PATH, adding /usr/local/bin temporarily"
  export PATH="/usr/local/bin:$PATH"
fi

# Verify install
echo "✅ Installed: $(buildkitd --version)"

# Cleanup
rm -rf "$TMPDIR"

echo "🔍 Detecting largest non-system volume..."

# Use consistent, POSIX-compatible df output
# Filter out ephemeral/system mounts
mapfile -t lines < <(
  df -B1 --output=target,avail,fstype 2>/dev/null | \
  tail -n +2 | \
  awk '$3 !~ /(tmpfs|devtmpfs|overlay|squashfs|efivarfs|proc|sysfs|cgroup|debugfs|rpc_pipefs|run)/ {print $0}' | \
  sort -k2 -n -r
)

if [ ${#lines[@]} -eq 0 ]; then
  echo "❌ No valid mounted volumes found."
  exit 1
fi

read -r mount avail fstype <<<"${lines[0]}"

avail_h=$(numfmt --to=iec <<<"$avail" 2>/dev/null || echo "$avail bytes")
echo "📦 Largest volume: $mount  (Free: $avail_h, Type: $fstype)"

# Export variable for use by parent script
echo "BUILD_VOLUME=$mount"

# Identify Buildkit root
if [ "$BUILD_VOLUME" = "/" ]; then
  BUILD_ROOT="/buildkit"
else
  BUILD_ROOT="$BUILD_VOLUME/buildkit"
fi

sudo mkdir -p "$BUILD_ROOT"
sudo chown root:root "$BUILD_ROOT"

echo "🧱 Using BuildKit root at: $BUILD_ROOT"