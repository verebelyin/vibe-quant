#!/bin/sh
# Pinned cccc (cognitive/cyclomatic complexity, github.com/moznion/cccc) for the test rule in
# CLAUDE.md § Tests prove outcomes. Downloads the release once (sha256-checked) into
# ~/.cache/vibe-quant/cccc-<version>/, then runs it with your args:
#   scripts/agents/cccc.sh vibe_quant/overfitting/dsr.py --table
set -eu
VERSION=1.9.2
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) TARGET=aarch64-apple-darwin
    SHA=8e8166931165add165dd7854b39a00c499087b45f566728367b49ad4117c069d ;;
  *) echo "cccc.sh: no pinned build for $(uname -s)-$(uname -m); add its sha256 here" >&2; exit 2 ;;
esac
DIR="${XDG_CACHE_HOME:-$HOME/.cache}/vibe-quant/cccc-$VERSION"
BIN="$DIR/cccc"
if [ ! -x "$BIN" ]; then
  mkdir -p "$DIR"
  TAR="$DIR/cccc.tar.gz"
  curl -fsSL -o "$TAR" "https://github.com/moznion/cccc/releases/download/v$VERSION/cccc-v$VERSION-$TARGET.tar.gz"
  echo "$SHA  $TAR" | shasum -a 256 -c - >/dev/null || { rm -f "$TAR"; echo "cccc.sh: sha256 mismatch" >&2; exit 2; }
  tar xzf "$TAR" -C "$DIR" cccc
  rm -f "$TAR"
fi
exec "$BIN" --no-config "$@"
