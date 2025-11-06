#!/usr/bin/env bash
set -euo pipefail

# Detect the largest mounted volume (by available space)
# Skips tmpfs, devtmpfs, overlay, squashfs, etc.

echo "🔍 Detecting largest non-system volume..."

# Collect filesystem info
# Use human-readable-free output but sort numerically by available blocks
mapfile -t lines < <(df -P -B1 --output=target,avail,fstype | \
  awk 'NR>1 && $3 !~ /(tmpfs|devtmpfs|overlay|squashfs|efivarfs|proc|sysfs)/ {print $0}' | \
  sort -k2 -n -r)

if [ ${#lines[@]} -eq 0 ]; then
  echo "❌ No valid mounted volumes found."
  exit 1
fi

# Parse the first line (largest available)
read -r mount avail fstype <<<"${lines[0]}"

avail_h=$(numfmt --to=iec <<<"$avail")
echo "📦 Largest volume: $mount  (Free: $avail_h, Type: $fstype)"

# Export for other scripts
echo "BUILD_VOLUME=$mount"
