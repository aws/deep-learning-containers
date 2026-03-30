#!/usr/bin/env bash
set -euo pipefail

# Detect embedded credentials, API keys, and internal tooling artifacts in DLC images
FAILED=0

# --- Credential files that should never exist in a production image ---
CREDENTIAL_FILES=(
  "$HOME/.aws/credentials"
  "$HOME/.aws/config"
  "$HOME/.git-credentials"
  "$HOME/.config/git/credentials"
  "$HOME/.netrc"
  "$HOME/.docker/config.json"
  "$HOME/.npmrc"
  "$HOME/.pypirc"
  "$HOME/.boto"
  "/etc/boto.cfg"
  # Hugging Face tokens (common in ML containers)
  "$HOME/.cache/huggingface/token"
  "$HOME/.huggingface/token"
)

for F in "${CREDENTIAL_FILES[@]}"; do
  if [ -f "$F" ]; then
    echo "FAIL: Credential file found: $F"
    FAILED=1
  fi
done

# .gitconfig is only a problem if it contains credentials or token extraheaders
if [ -f "$HOME/.gitconfig" ] && grep -qiE 'credential|extraheader' "$HOME/.gitconfig" 2>/dev/null; then
  echo "FAIL: $HOME/.gitconfig contains credential configuration"
  FAILED=1
fi

# .git directory should never be copied into a production image
# Note: /workdir is the host-mounted test repo in CI, so we skip it
for GIT_DIR in /.git "$HOME/.git" /opt/.git /src/.git; do
  if [ -d "$GIT_DIR" ]; then
    echo "FAIL: .git directory found at $GIT_DIR (source repo history leaked)"
    FAILED=1
  fi
done

# .env files may contain secrets
for DIR in / "$HOME" /opt; do
  for ENVFILE in "$DIR/.env" "$DIR/.env.local"; do
    if [ -f "$ENVFILE" ]; then
      echo "FAIL: Environment file found: $ENVFILE"
      FAILED=1
    fi
  done
done

# --- Environment variable leaks ---
# Check for AWS credential variables, GitHub Actions tokens, and ML platform keys
while IFS='=' read -r KEY _; do
  case "$KEY" in
    AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|\
    GITHUB_TOKEN|GH_TOKEN|ACTIONS_RUNTIME_TOKEN|ACTIONS_ID_TOKEN_REQUEST_TOKEN|\
    HF_TOKEN|HUGGING_FACE_HUB_TOKEN|\
    WANDB_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|\
    CODECOV_TOKEN|SNYK_TOKEN|SONAR_TOKEN|DOCKER_PASSWORD)
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
GREP_EXCLUDE=(--exclude-dir=venv --exclude-dir=conda --exclude-dir=lib --exclude-dir=site-packages)

for DIR in /etc /opt "$HOME"; do
  [ -d "$DIR" ] || continue
  while IFS= read -r F; do
    echo "FAIL: AWS access key pattern found in $F"
    FAILED=1
  done < <(grep -rlI 'AKIA[0-9A-Z]\{16\}' "$DIR" \
    "${GREP_EXCLUDE[@]}" 2>/dev/null || true)
done

# Scan for GitHub token patterns (ghp_, gho_, ghu_, ghs_, ghr_, github_pat_)
for DIR in /etc /opt "$HOME"; do
  [ -d "$DIR" ] || continue
  while IFS= read -r F; do
    echo "FAIL: GitHub token pattern found in $F"
    FAILED=1
  done < <(grep -rlIE 'gh[pousr]_[A-Za-z0-9]{36}|github_pat_' "$DIR" \
    "${GREP_EXCLUDE[@]}" 2>/dev/null || true)
done

# Scan for Hugging Face token patterns (hf_...)
for DIR in /etc /opt "$HOME"; do
  [ -d "$DIR" ] || continue
  while IFS= read -r F; do
    echo "FAIL: Hugging Face token pattern found in $F"
    FAILED=1
  done < <(grep -rlIE 'hf_[A-Za-z]{34}' "$DIR" \
    "${GREP_EXCLUDE[@]}" 2>/dev/null || true)
done

# Scan for private key headers outside standard certificate paths
# Exclude .ssh and ssh — build scripts generate SSH user keys (.ssh/id_rsa) for
# inter-container communication and host keys (/etc/ssh/ssh_host_*) for sshd
for DIR in /etc /opt "$HOME"; do
  [ -d "$DIR" ] || continue
  while IFS= read -r F; do
    echo "FAIL: Private key header found in $F"
    FAILED=1
  done < <(grep -rlI --exclude-dir=ssl --exclude-dir=.ssh --exclude-dir=ssh "${GREP_EXCLUDE[@]}" \
    -- '-----BEGIN.*PRIVATE KEY-----' "$DIR" \
    2>/dev/null || true)
done

# Scan pip.conf and .condarc for embedded credentials (URLs with user:pass@)
for CONF in /etc/pip.conf "$HOME/.pip/pip.conf" "$HOME/.config/pip/pip.conf" "$HOME/.condarc" /etc/conda/.condarc; do
  [ -f "$CONF" ] || continue
  if grep -qE '://[^/]*:[^/]*@' "$CONF" 2>/dev/null; then
    echo "FAIL: Embedded credentials in package config: $CONF"
    FAILED=1
  fi
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
