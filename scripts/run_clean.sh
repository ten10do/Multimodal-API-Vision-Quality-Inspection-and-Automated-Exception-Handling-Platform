#!/usr/bin/env bash
# Clean-environment runner for torch-dependent commands (Phase 7).
#
# The WorkBuddy CLI session injects variables (e.g. PYTHONPATH shims, MCP
# config JSONs) into the Bash environment. On this machine, importing torch
# inside that environment crashes with an access violation (0xC0000005).
# A minimal `env -i` environment (verified) runs torch fine. This wrapper
# rebuilds it. IVQC_* variables from the caller are forwarded.
#
# Usage:
#   IVQC_DATABASE_URL=... bash scripts/run_clean.sh python -m pytest ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv/Scripts"

# collect caller-provided IVQC_* vars to forward through env -i
FORWARD=()
while IFS= read -r line; do
  FORWARD+=("$line")
done < <(env | grep '^IVQC_' || true)

env -i \
  PATH="/c/Windows/System32:/c/Windows:/c/Windows/System32/Wbem:/c/Windows/System32/WindowsPowerShell/v1.0:/c/Program Files/Docker/Docker/resources/bin:$VENV" \
  SYSTEMROOT="C:\\Windows" \
  TEMP="C:\\Users\\EDY\\AppData\\Local\\Temp" \
  TMP="C:\\Users\\EDY\\AppData\\Local\\Temp" \
  USERNAME="EDY" \
  USERPROFILE="C:\\Users\\EDY" \
  HOMEDRIVE="C:" \
  HOMEPATH="\\Users\\EDY" \
  PATHEXT=".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC" \
  "${FORWARD[@]}" \
  "$@"
