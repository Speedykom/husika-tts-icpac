#!/bin/bash
set -euo pipefail

DEPLOY_DIR=${DEPLOY_DIR:-/srv/husika}
BACKUP_DIR=$DEPLOY_DIR/backups
ENV_FILE=$DEPLOY_DIR/.env

# Read one KEY=value from the deploy .env without executing it (see backup.sh).
env_value() {
  [ -f "$ENV_FILE" ] || return 0
  sed -n "s/^[[:space:]]*$1=//p" "$ENV_FILE" | tail -n 1 |
    sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

# Host-side live DB: ./data is bind-mounted to the container's data dir, so the
# file name follows DB_PATH, exactly as in backup.sh.
DB_PATH=${DB_PATH:-$(env_value DB_PATH)}
DB_PATH=${DB_PATH:-/app/data/husika.db}
LIVE_DB=$DEPLOY_DIR/data/$(basename "$DB_PATH")

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <backup_file>"
  echo ""
  echo "Available backups:"
  ls -lht "$BACKUP_DIR"/husika_*.db 2>/dev/null || echo "  No backups found in $BACKUP_DIR"
  exit 1
fi

BACKUP_FILE=$1

if [ ! -f "$BACKUP_FILE" ]; then
  echo "ERROR: file not found: $BACKUP_FILE"
  exit 1
fi

# Validate the backup before touching the live DB. This is the destructive step,
# so refuse a truncated or non-SQLite file rather than silently replacing good
# data. We validate up front so a bad file never triggers any downtime.
validate_sqlite() {
  local f=$1
  # Cheap header check: every valid SQLite DB starts with this 16-byte magic.
  if [ "$(head -c 15 "$f")" != "SQLite format 3" ]; then
    return 1
  fi
  # Deeper integrity check when a SQLite-capable tool is on the host.
  if command -v sqlite3 >/dev/null 2>&1; then
    [ "$(sqlite3 "$f" 'PRAGMA integrity_check')" = "ok" ]
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$f" <<'PY'
import sqlite3, sys
try:
    con = sqlite3.connect(sys.argv[1])
    ok = con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
  fi
}

# Check a scratch copy: opening a WAL-mode DB creates -wal/-shm sidecars, which
# we don't want left next to the backup file itself.
CHECK_DIR=$(mktemp -d)
trap 'rm -rf "$CHECK_DIR"' EXIT
cp "$BACKUP_FILE" "$CHECK_DIR/check.db"
if ! validate_sqlite "$CHECK_DIR/check.db"; then
  echo "ERROR: $BACKUP_FILE failed SQLite validation — refusing to restore."
  exit 1
fi

echo "Restoring from: $BACKUP_FILE"
echo "This will restart the app. Press Ctrl+C within 5 seconds to cancel."
sleep 5

# Intentionally brings the whole stack down (app + Caddy), so the public site
# returns 502 for the duration of the restore. This is an accepted maintenance
# window: it guarantees no process holds husika.db open while we swap the file.
docker compose -f "$DEPLOY_DIR/docker-compose.yml" down

# Keep the current DB (plus any WAL/SHM sidecars) so a mistaken restore can be
# undone. Named outside the husika_*.db pattern so backup.sh never prunes it.
if [ -f "$LIVE_DB" ]; then
  SAFETY_DIR=$BACKUP_DIR/pre-restore_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$SAFETY_DIR"
  cp -p "$LIVE_DB" "$LIVE_DB"-wal "$LIVE_DB"-shm "$SAFETY_DIR"/ 2>/dev/null || true
  echo "Previous database saved to: $SAFETY_DIR"
fi

# The app runs SQLite in WAL mode. A -wal/-shm left by the old DB (e.g. after an
# unclean shutdown) would be replayed on top of the restored file and corrupt
# it, so remove them before swapping the file in.
rm -f "$LIVE_DB"-wal "$LIVE_DB"-shm
cp "$BACKUP_FILE" "$LIVE_DB"
docker compose -f "$DEPLOY_DIR/docker-compose.yml" up -d

# `up -d` returns as soon as containers start, not once the app is serving.
# Poll the compose healthcheck so "Restore complete" only prints when the app
# is actually healthy. Cap the wait so a failed start errors out instead of
# hanging (healthcheck: 30s start_period + 30s interval, so ~120s is ample).
echo "Waiting for the app to become healthy..."
app_cid=$(docker compose -f "$DEPLOY_DIR/docker-compose.yml" ps -q app || true)
health=""
for _ in $(seq 1 60); do
  health=$(docker inspect --format '{{.State.Health.Status}}' "$app_cid" 2>/dev/null || true)
  [ "$health" = "healthy" ] && break
  [ "$health" = "unhealthy" ] && break
  sleep 2
done

if [ "$health" != "healthy" ]; then
  echo "ERROR: app did not become healthy after restore (status: ${health:-unknown})."
  echo "Inspect with: docker compose -f $DEPLOY_DIR/docker-compose.yml logs app"
  exit 1
fi

echo "[$(date)] Restore complete from $BACKUP_FILE"
