#!/usr/bin/env bash
# Install repo-managed git hooks (no global git config required beyond this repo).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "install-git-hooks: not inside a git repository" >&2
  exit 1
fi

HOOKS_PATH=".githooks"
if [[ ! -d "$HOOKS_PATH" ]]; then
  echo "install-git-hooks: missing $HOOKS_PATH/" >&2
  exit 1
fi

chmod +x "$HOOKS_PATH"/pre-commit "$HOOKS_PATH"/pre-push

# Prefer core.hooksPath so hooks stay versioned in-tree.
git config --local core.hooksPath "$HOOKS_PATH"

echo "Installed git hooks via core.hooksPath=$HOOKS_PATH"
echo "  pre-commit: cargo fmt --check"
echo "  pre-push:   fmt + clippy + test + doc (CI parity)"
echo
echo "Bypass with --no-verify when needed."
