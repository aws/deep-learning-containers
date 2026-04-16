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

# Compute keepBytes as 80% of available space
KEEP_BYTES=$(awk -v avail="$avail" 'BEGIN {printf "%.0f", avail * 0.8}')
KEEP_HUMAN=$(numfmt --to=iec <<<"$KEEP_BYTES")

echo "🧮 Setting BuildKit GC limit to ~80% of free space: $KEEP_HUMAN"

# -----------------------------
# Step 3. Generate and write configuration (diff-based drift detection)
# -----------------------------
sudo mkdir -p /etc/buildkit
CONFIG_PATH="/etc/buildkit/buildkitd.toml"

TMP_CONFIG=$(mktemp)
sudo chmod 644 "$TMP_CONFIG"
cat <<EOF > "$TMP_CONFIG"
[worker.oci]
  enabled = true
  gc = true
  root = "$BUILD_ROOT"
  networkMode = "host"

[gc]
  enabled = true
  defaultKeepStorage = "$KEEP_HUMAN"
  [[gc.policy]]
    keepDuration = "720h"    # 30 days
    filters = ["type==regular"]
EOF

if [ ! -f "$CONFIG_PATH" ]; then
  echo "🆕 No existing config found — creating $CONFIG_PATH"
  sudo cp "$TMP_CONFIG" "$CONFIG_PATH"
  CONFIG_CHANGED=true
else
  if ! sudo diff -qwB "$TMP_CONFIG" "$CONFIG_PATH" >/dev/null 2>&1; then
    echo "⚠️ Config drift detected — updating $CONFIG_PATH"
    sudo cp "$TMP_CONFIG" "$CONFIG_PATH"
    CONFIG_CHANGED=true
  else
    echo "✅ Config unchanged."
    CONFIG_CHANGED=false
  fi
fi

rm -f "$TMP_CONFIG"

# -----------------------------
# Step 4. Ensure systemd service exists
# -----------------------------
SERVICE_FILE="/etc/systemd/system/buildkitd.service"

if [ ! -f "$SERVICE_FILE" ]; then
  echo "🧾 Installing BuildKit systemd service..."
  sudo tee "$SERVICE_FILE" >/dev/null <<'EOF'
[Unit]
Description=BuildKit Daemon
After=network.target

[Service]
ExecStart=/usr/local/bin/buildkitd --config /etc/buildkit/buildkitd.toml
Restart=always
LimitNOFILE=1048576
OOMScoreAdjust=-500

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable buildkitd
else
  echo "✅ Systemd unit already exists."
fi

# -----------------------------
# Step 5. Check daemon health and config drift
# -----------------------------
echo "🩺 Checking BuildKit daemon health..."

if systemctl is-active --quiet buildkitd; then
  if [ "$CONFIG_CHANGED" = true ]; then
    echo "🔄 buildkitd active but config changed — reloading..."
    sudo systemctl daemon-reload
    sudo systemctl restart buildkitd
  else
    echo "✅ buildkitd active and config unchanged. Skipping restart."
  fi
else
  echo "🔄 buildkitd not running or inactive — starting..."
  sudo systemctl daemon-reload
  sudo systemctl restart buildkitd
fi

# -----------------------------
# Step 6. Verify status
# -----------------------------
sleep 3
sudo systemctl --no-pager status buildkitd | head -n 15
echo "🔍 Verifying workers..."
buildctl debug workers || (echo "⚠️ buildctl failed to connect" && exit 1)

echo "✅ BuildKit daemon setup complete!"