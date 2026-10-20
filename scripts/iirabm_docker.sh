#!/usr/bin/env bash
# Run the engine CLI inside the pinned reference container, reading JSON Lines on stdin.
# Use as IIRABM_BIN for local runs on a Mac:  export IIRABM_BIN=scripts/iirabm_docker.sh
#   IIRABM_PLATFORM  linux/arm64 (default, fast; byte-identical to amd64 on the golden files)
#                    or linux/amd64 (the reference platform; emulated on Apple Silicon)
# The binary must have been built for that platform:
#   engine/tools/ref.sh --platform linux/arm64 -- make cli
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${ENGINE_DIR:-$HERE/engine}"
ENGINE="$(cd "$ENGINE" && pwd)"
PLATFORM="${IIRABM_PLATFORM:-linux/arm64}"
case "$PLATFORM" in
  linux/arm64) ARCH=aarch64 ;;
  linux/amd64) ARCH=x86_64 ;;
  *) echo "unsupported IIRABM_PLATFORM $PLATFORM" >&2; exit 2 ;;
esac
IMAGE="gcc:13.3.0-bookworm@sha256:1d71f0f3450214bef38fe09e6f610fb6cca90cf97b43f4ce845bfc32a4168818"
exec docker run --rm -i --platform "$PLATFORM" -v "$ENGINE":/engine:ro "$IMAGE" \
  "/engine/build/Linux-$ARCH/iirabm_cli" "$@"
