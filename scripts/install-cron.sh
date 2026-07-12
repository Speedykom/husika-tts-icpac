#!/bin/bash
# Run on the server only. Paths are hardcoded to /srv/husika (where the compose
# stack lives) and it writes to /etc/cron.d and /var/log — do not run locally.
set -euo pipefail

# Single source of truth for the deploy dir and container name in this file.
# CONTAINER matches the pinned Compose project name (name: husika -> husika-app-1)
# and backup.sh's own defaults.
DEPLOY_DIR=/srv/husika
CONTAINER=husika-app-1

chmod +x "$DEPLOY_DIR/scripts/backup.sh" "$DEPLOY_DIR/scripts/restore.sh"

# Install daily cron job at 2am
cat > /etc/cron.d/husika-backup << EOF
# Daily backup of husika TTS database at 2:00 AM
0 2 * * * root DEPLOY_DIR=$DEPLOY_DIR CONTAINER=$CONTAINER $DEPLOY_DIR/scripts/backup.sh >> /var/log/husika-backup.log 2>&1
EOF

chmod 644 /etc/cron.d/husika-backup
echo "Cron job installed. Daily backup runs at 2:00 AM."
echo "Logs: /var/log/husika-backup.log"
echo "Backups stored in: $DEPLOY_DIR/backups/"