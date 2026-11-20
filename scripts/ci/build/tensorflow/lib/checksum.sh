#!/usr/bin/env bash
# Checksum sidecar for the cached TF artifact (sourced by pre_build.sh and
# post_build.sh).
#
# The Dockerfile's source-build stage writes <artifact>.sha256 (the bare hex
# digest) right after the binary passes the cuobjdump GPU-target check, and the
# export stage ships it next to the artifact. An artifact counts as complete only
# when its digest matches the sidecar, so a truncated export or download is never
# uploaded or used.

# artifact_verified <artifact path>: 0 if <artifact>.sha256 exists and matches.
artifact_verified() {
  local file="$1" want got
  [[ -f "$file" && -f "${file}.sha256" ]] || return 1
  want=$(tr -d '[:space:]' < "${file}.sha256")
  got=$(sha256sum "$file" | cut -d' ' -f1)
  [[ -n "$want" && "$want" == "$got" ]]
}
