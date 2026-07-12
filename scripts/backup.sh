#!/bin/bash
set -euo pipefail

DEPLOY_DIR=${DEPLOY_DIR:-/srv/husika}
CONTAINER=${CONTAINER:-husika-app-1}
BACKUP_DIR=$DEPLOY_DIR/backups
DB_PATH=$DEPLOY_DIR/data/husika.db
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE=$BACKUP_DIR/husika_$TIMESTAMP.db
TMP_DB=$DEPLOY_DIR/data/backup_tmp.db
KEEP_DAYS=7

# Ensure a stale temp copy is never left behind if the backup fails partway.
trap 'rm -f "$TMP_DB"' EXIT

mkdir -p "$BACKUP_DIR"

if [ ! -f "$DB_PATH" ]; then
  echo "ERROR: database not found at $DB_PATH"
  exit 1
fi

# Use SQLite backup API for a consistent copy (safe while app is running)
docker exec "$CONTAINER" /app/.venv/bin/python3 -c "
import sqlite3, sys
src = sqlite3.connect('/app/data/husika.db')
dst = sqlite3.connect('/app/data/backup_tmp.db')
src.backup(dst)
src.close()
dst.close()
"

mv "$TMP_DB" "$BACKUP_FILE"

echo "[$(date)] Backup saved: $BACKUP_FILE"

# Remove backups older than KEEP_DAYS
find "$BACKUP_DIR" -name "husika_*.db" -mtime +$KEEP_DAYS -delete
echo "[$(date)] Cleaned up backups older than $KEEP_DAYS days"
