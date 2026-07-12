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

echo "Restoring from: $BACKUP_FILE"
echo "This will restart the app. Press Ctrl+C within 5 seconds to cancel."
sleep 5

# Intentionally brings the whole stack down (app + Caddy), so the public site
# returns 502 for the duration of the restore. This is an accepted maintenance
# window: it guarantees no process holds husika.db open while we swap the file.
docker compose -f "$DEPLOY_DIR/docker-compose.yml" down
cp "$BACKUP_FILE" "$DEPLOY_DIR/data/husika.db"
docker compose -f "$DEPLOY_DIR/docker-compose.yml" up -d

echo "[$(date)] Restore complete from $BACKUP_FILE"