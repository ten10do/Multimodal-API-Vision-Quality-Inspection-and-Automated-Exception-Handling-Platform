#!/usr/bin/env bash
# One-command demo (10G).
#
# Starts: PostgreSQL (Docker) -> simulators (PLC/MES/OPC UA) -> inference
# (host) -> backend (host, industrial mode) -> demo seed -> frontend (host).
#
# Principle: infrastructure in Docker, GPU inference on the host (Windows +
# RTX); we do not force GPU into a container (stability first).
#
# Usage:  bash scripts/demo_up.sh        (or: ./scripts/demo_up.sh)

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY=".venv/Scripts/python.exe"
# Node / npm: honor $NODE_JS / $NPM overrides, otherwise auto-locate common
# installs. No hard-coded user paths (works on any machine).
NODE_JS="${NODE_JS:-}"
if [ -z "$NODE_JS" ]; then
  for cand in \
    "${USERPROFILE//\\//}/.workbuddy/binaries/node/versions"/*/node.exe \
    "${HOME//\\//}/.workbuddy/binaries/node/versions"/*/node.exe \
    "C:/Program Files/nodejs/node.exe"; do
    [ -x "$cand" ] && NODE_JS="$cand" && break
  done
  [ -z "$NODE_JS" ] && NODE_JS="node"
fi
NPM="${NPM:-}"
if [ -z "$NPM" ]; then
  for cand in \
    "$(dirname "$NODE_JS")/../node_modules/npm/bin/npm-cli.js" \
    "C:/Program Files/nodejs/node_modules/npm/bin/npm-cli.js"; do
    [ -e "$cand" ] && NPM="$cand" && break
  done
  [ -z "$NPM" ] && NPM="$(command -v npm 2>/dev/null || echo '')"
fi
DB_URL="postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5433/industrialvision_test"

port_up() { # $1 = port
  "$PY" -c "import socket,sys
try:
    s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',$1)); s.close(); sys.exit(0)
except Exception: sys.exit(1)" 2>/dev/null
}

echo "== 1/7 PostgreSQL (Docker) =="
if port_up 5433; then echo "  postgres already up"; else
  (docker compose up -d postgres >/dev/null 2>&1 || "/c/Program Files/Docker/Docker/resources/bin/docker" compose up -d postgres >/dev/null 2>&1)
  for i in $(seq 1 20); do port_up 5433 && break; sleep 2; done
  echo "  postgres up"
fi

echo "== 2/7 Simulators (PLC HTTP 8501, MES 8502, OPC UA 8503) =="
for p in 8501 8502 8503; do port_up "$p" || true; done
if ! port_up 8501; then ( "$PY" -m simulator.plc_simulator > /tmp/demo-plc.log 2>&1 & ); fi
if ! port_up 8502; then ( "$PY" -m simulator.mes_simulator > /tmp/demo-mes.log 2>&1 & ); fi
if ! port_up 8503; then ( bash scripts/run_clean.sh "$PY" -m simulator.opcua_plc_server > /tmp/demo-opcua.log 2>&1 & ); fi
sleep 4
echo "  simulators ready"

echo "== 3/7 Inference service (host, 8100) =="
if ! port_up 8100; then
  ( bash scripts/run_clean.sh "$PY" -m uvicorn inference_app.api:app --host 127.0.0.1 --port 8100 --app-dir inference-service > /tmp/demo-infer.log 2>&1 & )
  for i in $(seq 1 60); do port_up 8100 && break; sleep 2; done
fi
echo "  inference up (verify /ready)"

echo "== 4/7 Backend (host, 8000, industrial mode) =="
if ! port_up 8000; then
  ( IVQC_DATABASE_URL="$DB_URL" IVQC_INFERENCE_SERVICE_URL="http://127.0.0.1:8100" \
      IVQC_PLC_ENABLED=true IVQC_MES_ENABLED=true IVQC_LLM_PROVIDER=fake \
      bash scripts/run_clean.sh "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend > /tmp/demo-backend.log 2>&1 & )
  for i in $(seq 1 40); do port_up 8000 && break; sleep 2; done
fi
echo "  backend up"

echo "== 5/7 Demo seed (deterministic fixture) =="
IVQC_DATABASE_URL="$DB_URL" bash scripts/run_clean.sh "$PY" scripts/demo_seed.py

echo "== 6/7 Frontend (Vite, 5173) =="
if ! port_up 5173; then
  ( cd frontend && NODE_OPTIONS="" "$NODE_JS" "$NPM" run dev -- --host 127.0.0.1 --port 5173 > /tmp/demo-vite.log 2>&1 & )
  for i in $(seq 1 30); do port_up 5173 && break; sleep 2; done
fi
echo "  frontend up"

echo "== 7/7 Health check =="
bash scripts/run_clean.sh "$PY" scripts/health_check.py | tail -12

echo
echo "Demo ready:  http://127.0.0.1:5173"
echo "Backend API: http://127.0.0.1:8000/ready   (industrial mode: PLC/MES enabled)"
echo "Stop demo:   stop the background processes or close the terminals."
