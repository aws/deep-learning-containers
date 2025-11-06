#!/usr/bin/env bash
set -euo pipefail

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
