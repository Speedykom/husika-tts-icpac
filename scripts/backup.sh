#!/bin/bash
set -euo pipefail

DEPLOY_DIR=${DEPLOY_DIR:-/srv/husika}

ENV_FILE=$DEPLOY_DIR/.env

# Read one KEY=value from the deploy .env without executing it. The file is in
# docker compose's format, not shell: JSON model maps or secrets containing
# spaces, quotes or `$` would break (or run) under `source`, and with `set -e`
# that would silently stop the nightly backup.
env_value() {
  [ -f "$ENV_FILE" ] || return 0
  sed -n "s/^[[:space:]]*$1=//p" "$ENV_FILE" | tail -n 1 |
    sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

# Precedence: variable already set in the environment > deploy .env > default.
CONTAINER=${CONTAINER:-$(env_value CONTAINER)}
CONTAINER=${CONTAINER:-husika-app-1}
# In-container DB path, matching docker-compose.yml's DB_PATH.
DB_PATH=${DB_PATH:-$(env_value DB_PATH)}
DB_PATH=${DB_PATH:-/app/data/husika.db}
KEEP_DAYS=${BACKUP_KEEP_DAYS:-$(env_value BACKUP_KEEP_DAYS)}
KEEP_DAYS=${KEEP_DAYS:-7}

# Host-side paths. ./data is bind-mounted to the container's data dir, so the
# live DB and the temp copy the container writes both appear under ./data.
DB_NAME=$(basename "$DB_PATH")
DATA_DIR=$DEPLOY_DIR/data
BACKUP_DIR=$DEPLOY_DIR/backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE=$BACKUP_DIR/husika_$TIMESTAMP.db
TMP_DB=$DATA_DIR/backup_tmp.db
HOST_DB_PATH=$DATA_DIR/$DB_NAME
CONTAINER_TMP_DB=$(dirname "$DB_PATH")/backup_tmp.db

# Ensure a stale temp copy is never left behind if the backup fails partway.
trap 'rm -f "$TMP_DB"' EXIT

mkdir -p "$BACKUP_DIR"

if [ ! -f "$HOST_DB_PATH" ]; then
  echo "ERROR: database not found at $HOST_DB_PATH"
  exit 1
fi

# Use SQLite backup API for a consistent copy (safe while app is running)
docker exec "$CONTAINER" /app/.venv/bin/python3 -c "
import sqlite3
src = sqlite3.connect('$DB_PATH')
dst = sqlite3.connect('$CONTAINER_TMP_DB')
src.backup(dst)
src.close()
dst.close()
"

mv "$TMP_DB" "$BACKUP_FILE"

echo "[$(date)] Backup saved: $BACKUP_FILE"

# Remove backups older than KEEP_DAYS
find "$BACKUP_DIR" -name "husika_*.db" -mtime +"$KEEP_DAYS" -delete
echo "[$(date)] Cleaned up backups older than $KEEP_DAYS days"
