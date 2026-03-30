#!/usr/bin/env bash
set -euo pipefail

# Detect embedded credentials, API keys, and internal tooling artifacts in DLC images
FAILED=0

# --- Credential files that should never exist in a production image ---
CREDENTIAL_FILES=(
  "$HOME/.aws/credentials"
  "$HOME/.aws/config"
  "$HOME/.git-credentials"
  "$HOME/.netrc"
  "$HOME/.docker/config.json"
  "$HOME/.ssh/id_rsa"
  "$HOME/.ssh/id_ed25519"
  "$HOME/.ssh/id_ecdsa"
  "$HOME/.ssh/id_dsa"
  "$HOME/.npmrc"
  "$HOME/.pypirc"
  "$HOME/.boto"
  "/etc/boto.cfg"
)

for F in "${CREDENTIAL_FILES[@]}"; do
  if [ -f "$F" ]; then
    echo "FAIL: Credential file found: $F"
    FAILED=1
  fi
done

# .gitconfig is only a problem if it contains credentials
if [ -f "$HOME/.gitconfig" ] && grep -qi 'credential' "$HOME/.gitconfig" 2>/dev/null; then
  echo "FAIL: $HOME/.gitconfig contains credential configuration"
  FAILED=1
fi

# --- Environment variable leaks ---
# Check for AWS credential variables and suspicious secret-like variables
while IFS='=' read -r KEY _; do
  case "$KEY" in
    AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)
      echo "FAIL: Sensitive environment variable set: $KEY"
      FAILED=1
      ;;
  esac
done < <(env)

while IFS='=' read -r KEY _; do
  # Skip known safe variables
  case "$KEY" in
    DBUS_SESSION_BUS_ADDRESS) continue ;;
    SAGEMAKER_*|DLC_*)        continue ;;
  esac
  case "$KEY" in
    *_API_KEY|*_SECRET|*_TOKEN|*_PASSWORD)
      echo "FAIL: Suspicious environment variable set: $KEY"
      FAILED=1
      ;;
  esac
done < <(env)

# --- Filesystem scan for embedded secrets ---
# Scan config directories for AWS access key patterns (AKIA...)
# Use -I to skip binary files; exclude large package directories to avoid slow scans
for DIR in /etc /opt "$HOME"; do
  [ -d "$DIR" ] || continue
  while IFS= read -r F; do
    echo "FAIL: AWS access key pattern found in $F"
    FAILED=1
  done < <(grep -rlI 'AKIA[0-9A-Z]\{16\}' "$DIR" \
    --exclude-dir=venv --exclude-dir=conda --exclude-dir=lib \
    --exclude-dir=site-packages \
    2>/dev/null || true)
done

# Scan for private key headers outside standard certificate paths
for DIR in /etc /opt "$HOME"; do
  [ -d "$DIR" ] || continue
  while IFS= read -r F; do
    echo "FAIL: Private key header found in $F"
    FAILED=1
  done < <(grep -rlI -- '-----BEGIN.*PRIVATE KEY-----' "$DIR" \
    --exclude-dir=ssl --exclude-dir=venv --exclude-dir=conda \
    --exclude-dir=lib --exclude-dir=site-packages \
    2>/dev/null || true)
done

# --- Internal tooling artifacts that should not ship ---
INTERNAL_PATHS=(
  "/root/.brazil"
  "/root/.toolbox"
  "$HOME/.brazil"
  "$HOME/.toolbox"
  "/apollo"
  "/workplace"
)

for P in "${INTERNAL_PATHS[@]}"; do
  # Check for both exact paths and glob matches (e.g. .brazil*)
  for MATCH in "$P" "$P"*; do
    if [ -e "$MATCH" ]; then
      echo "FAIL: Internal tooling artifact found: $MATCH"
      FAILED=1
    fi
  done
done

# Check for internal tools in bin directories
for BIN_DIR in /usr/local/bin /usr/bin; do
  for PATTERN in midway mwinit; do
    for F in "$BIN_DIR"/*"$PATTERN"*; do
      [ -e "$F" ] || continue
      echo "FAIL: Internal tool found: $F"
      FAILED=1
    done
  done
done

exit $FAILED
