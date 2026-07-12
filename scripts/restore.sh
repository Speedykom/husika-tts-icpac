#!/bin/bash
set -euo pipefail

DEPLOY_DIR=${DEPLOY_DIR:-/srv/husika}
BACKUP_DIR=$DEPLOY_DIR/backups

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

if ! validate_sqlite "$BACKUP_FILE"; then
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
cp "$BACKUP_FILE" "$DEPLOY_DIR/data/husika.db"
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
