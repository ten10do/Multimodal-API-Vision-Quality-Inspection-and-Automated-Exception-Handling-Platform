"""Reproducible provisioning of the Docker-hosted test database.

Usage:
    .venv/Scripts/python.exe scripts/prepare_test_db.py [--if-missing]

Steps:
1. verify the Docker PostgreSQL container is reachable on host 127.0.0.1:5433
2. create industrialvision_test inside the container (only when missing,
   unless --if-missing is given, in which case the DB is dropped first so the
   migration is exercised from an empty test DB)
3. run `alembic upgrade head` against it
4. run the idempotent quality-rule seed

Guards:
- touches only industrialvision_test
- never drops/truncates industrialvision_dev
- never touches the Windows native PostgreSQL (127.0.0.1:5432)
- fails fast with a clear message when the Docker PostgreSQL is unreachable
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
DOCKER = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe")

HOST = "127.0.0.1"
PORT = 5433
TEST_DB = "industrialvision_test"
TEST_URL = f"postgresql+asyncpg://vision_qc:vision_qc@{HOST}:{PORT}/{TEST_DB}"
SYNC_DSN = f"postgresql://vision_qc:vision_qc@{HOST}:{PORT}/postgres"


def _compose_healthy() -> bool:
    probe = subprocess.run(
        [str(DOCKER), "compose", "ps", "--format", "{{.Status}}"],
        capture_output=True, text=True, timeout=30,
    )
    return "healthy" in probe.stdout


def _docker_db_exists(name: str) -> bool:
    r = subprocess.run(
        [str(DOCKER), "compose", "exec", "-T", "postgres", "psql", "-U", "vision_qc", "-d", "postgres",
         "-tAc", f"SELECT 1 FROM pg_database WHERE datname='{name}';"],
        capture_output=True, text=True, timeout=30,
    )
    return "1" in r.stdout


def _create_test_db() -> None:
    if not _docker_db_exists(TEST_DB):
        subprocess.run(
            [str(DOCKER), "compose", "exec", "-T", "postgres", "createdb", "-U", "vision_qc", TEST_DB],
            check=True, capture_output=True, timeout=60,
        )
        print(f"created {TEST_DB} in docker postgres ({HOST}:{PORT})")


def _recreate_test_db() -> None:
    subprocess.run(
        [str(DOCKER), "compose", "exec", "-T", "postgres", "dropdb", "-U", "vision_qc", "--if-exists", TEST_DB],
        capture_output=True, timeout=60,
    )
    subprocess.run(
        [str(DOCKER), "compose", "exec", "-T", "postgres", "createdb", "-U", "vision_qc", TEST_DB],
        check=True, capture_output=True, timeout=60,
    )
    print(f"recreated {TEST_DB} from empty")


def _alembic_upgrade() -> None:
    env = {**os.environ, "IVQC_DATABASE_URL": TEST_URL}
    r = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        print(r.stdout[-2000:], file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("alembic upgrade head failed against the test database")
    print("alembic upgrade head -> ok")


def _seed_rules() -> None:
    env = {**os.environ, "IVQC_DATABASE_URL": TEST_URL}
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "seed_quality_rules.py")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise SystemExit(f"seed failed: {r.stdout[-1000:]}{r.stderr[-1000:]}")
    print(r.stdout.strip())


def prepare_test_db(recreate: bool = False) -> str:
    """Provision the Docker-hosted test DB. Returns the async SQLAlchemy URL."""
    if not _compose_healthy():
        raise SystemExit(
            "Docker PostgreSQL container is not healthy. Start it first with:\n"
            "  docker compose up -d postgres\n"
            "then re-run this script or the integration tests."
        )
    if recreate:
        _recreate_test_db()
    else:
        _create_test_db()
    _alembic_upgrade()
    _seed_rules()
    return TEST_URL


def main() -> None:
    p = argparse.ArgumentParser(description="prepare the Docker-hosted integration test database")
    p.add_argument("--recreate", action="store_true", help="drop and recreate the test DB from empty before migrating")
    args = p.parse_args()
    url = prepare_test_db(recreate=args.recreate)
    print(f"test db ready: {url}")


if __name__ == "__main__":
    main()
