#!/usr/bin/env bash
set -euo pipefail

echo "🚀 BuildKit Host Bootstrap"

# -----------------------------
# Step 1. Detect or install BuildKit
# -----------------------------
if ! command -v buildkitd >/dev/null 2>&1; then
  echo "🔍 BuildKit not found, installing latest release..."

  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64)   ARCH="amd64" ;;
    aarch64)  ARCH="arm64" ;;
    *) echo "❌ Unsupported arch: $ARCH" >&2; exit 1 ;;
  esac

  LATEST_TAG=$(curl -sI https://github.com/moby/buildkit/releases/latest | \
               grep -i location | sed 's#.*/##' | tr -d '\r')

  if [ -z "$LATEST_TAG" ]; then
    echo "❌ Failed to get latest BuildKit release tag."
    exit 1
  fi

  echo "📦 Installing BuildKit $LATEST_TAG for $ARCH..."
  TMPDIR=$(mktemp -d)
  URL="https://github.com/moby/buildkit/releases/download/${LATEST_TAG}/buildkit-${LATEST_TAG}.linux-${ARCH}.tar.gz"
  curl -L "$URL" -o "$TMPDIR/buildkit.tar.gz"
  sudo tar -C /usr/local -xzf "$TMPDIR/buildkit.tar.gz"
  rm -rf "$TMPDIR"
else
  echo "✅ BuildKit already installed: $(buildkitd --version)"
fi

# -----------------------------
# Step 2. Detect the largest volume
# -----------------------------
echo "🔍 Detecting large volume for BuildKit cache..."

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
echo "📦 Largest volume: $mount (Free: $avail_h, Type: $fstype)"

if [ "$mount" = "/" ]; then
  BUILD_ROOT="/buildkit"
else
  BUILD_ROOT="$mount/buildkit"
fi

sudo mkdir -p "$BUILD_ROOT"
sudo chown root:root "$BUILD_ROOT"
echo "🧱 Using BuildKit root at: $BUILD_ROOT"

# -----------------------------
# Step 3. Write configuration
# -----------------------------
sudo mkdir -p /etc/buildkit

cat <<EOF | sudo tee /etc/buildkit/buildkitd.toml >/dev/null
[worker.oci]
  enabled = true
  gc = true
  root = "$BUILD_ROOT"

[gc]
  enabled = true
  defaultKeepStorage = "9TB"
  [[gc.policy]]
    keepDuration = "720h"    # 30 days
    keepBytes = "9TB"
    filters = ["type==regular"]
EOF

echo "✅ Wrote /etc/buildkit/buildkitd.toml"

# -----------------------------
# Step 4. Install systemd service
# -----------------------------
SERVICE_FILE="/etc/systemd/system/buildkitd.service"
sudo tee "$SERVICE_FILE" >/dev/null <<'EOF'
[Unit]
Description=BuildKit Daemon
After=network.target

[Service]
ExecStart=/usr/local/bin/buildkitd --config /etc/buildkit/buildkitd.toml --addr tcp://127.0.0.1:1234
Restart=always
LimitNOFILE=1048576
OOMScoreAdjust=-500

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now buildkitd

# -----------------------------
# Step 5. Verify daemon
# -----------------------------
echo "🩺 Checking BuildKit daemon status..."
sleep 3
sudo systemctl --no-pager status buildkitd || true

echo "🔍 Verifying workers..."
buildctl debug workers

echo "✅ BuildKit daemon setup complete!"
