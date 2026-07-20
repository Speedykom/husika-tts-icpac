#!/bin/bash
set -euo pipefail

DEPLOY_DIR=${DEPLOY_DIR:-/srv/husika}

ENV_FILE=$DEPLOY_DIR/.env
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  set +a
fi

CONTAINER=${CONTAINER:-husika-app-1}
# In-container DB path, matching docker-compose.yml's DB_PATH.
DB_PATH=${DB_PATH:-/app/data/husika.db}
KEEP_DAYS=${BACKUP_KEEP_DAYS:-7}

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
find "$BACKUP_DIR" -name "husika_*.db" -mtime +$KEEP_DAYS -delete
echo "[$(date)] Cleaned up backups older than $KEEP_DAYS days"
