#!/usr/bin/env bash
set -euo pipefail

# Container filesystem sanity checks
FAILED=0

# --- Stray .py artifacts in key directories ---
for DIR in /tmp /var/tmp "$HOME" /; do
  for F in "$DIR"/.* "$DIR"/*; do
    [ -e "$F" ] || continue
    NAME="${F##*/}"
    if [[ "$NAME" == *.py ]]; then
      echo "FAIL: Stray .py artifact $NAME found in $DIR"
      FAILED=1
    fi
  done
done

# --- /var/tmp must be empty ---
for F in /var/tmp/.* /var/tmp/*; do
  [ -e "$F" ] || continue
  NAME="${F##*/}"
  case "$NAME" in
    .|..) continue ;;
  esac
  echo "FAIL: /var/tmp is not empty: $NAME"
  FAILED=1
done

# --- /tmp should only contain expected files ---
for F in /tmp/.* /tmp/*; do
  [ -e "$F" ] || continue
  NAME="${F##*/}"
  case "$NAME" in
    .|..)         continue ;;
    .*)           continue ;;  # hidden files
    *system*)     continue ;;
    *System*)     continue ;;
    *dkms*)       continue ;;
    *hsperfdata*) continue ;;
  esac
  echo "FAIL: Unexpected file in /tmp: $NAME"
  FAILED=1
done

# --- Cache dir should only contain pip files ---
CACHE_DIR="$HOME/.cache"
if [ -d "$CACHE_DIR" ]; then
  for F in "$CACHE_DIR"/.* "$CACHE_DIR"/*; do
    [ -e "$F" ] || continue
    NAME="${F##*/}"
    case "$NAME" in
      .|..) continue ;;
      pip*) continue ;;
    esac
    echo "FAIL: Unexpected file in cache dir: $NAME"
    FAILED=1
  done
fi

# --- No .viminfo or non-empty .bash_history ---
if [ -f "$HOME/.viminfo" ]; then
  echo "FAIL: .viminfo exists"
  FAILED=1
fi
if [ -f "$HOME/.bash_history" ] && [ -s "$HOME/.bash_history" ]; then
  echo "FAIL: .bash_history contains history"
  FAILED=1
fi

# --- History files must not predate container boot ---
if [ -f /proc/uptime ]; then
  UPTIME=$(awk '{printf "%d", $1}' /proc/uptime)
  BOOT_TIME=$(( $(date +%s) - UPTIME ))

  for F in "$HOME"/*history*; do
    [ -f "$F" ] || continue
    MTIME=$(stat -c %Y "$F")
    if [ "$MTIME" -lt "$BOOT_TIME" ]; then
      echo "FAIL: $F was modified before container boot"
      FAILED=1
    fi
  done
fi

exit $FAILED
