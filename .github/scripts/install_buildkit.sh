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
