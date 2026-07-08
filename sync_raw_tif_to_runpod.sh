#!/usr/bin/env bash
set -euo pipefail

LOCAL_SRC="/Users/byepesg/Documents/Kumanday/IA/projects/EEG/Code/raw_tif"
REMOTE_HOST="runpod-bendr"
REMOTE_DIR="/workspace/EEG-Brain-DA/raw_tif"
SSH_KEY="$HOME/.ssh/id_ed25519"

if [ ! -d "$LOCAL_SRC" ]; then
  echo "Local source folder not found: $LOCAL_SRC"
  exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
  echo "SSH key not found: $SSH_KEY"
  exit 1
fi

ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"

tar -czf - -C "$(dirname "$LOCAL_SRC")" "$(basename "$LOCAL_SRC")" \
  | ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" "$REMOTE_HOST" \
    "mkdir -p '$REMOTE_DIR' && tar -xzf - -C '/workspace/EEG-Brain-DA'"

echo "Transfer completed. Files are in $REMOTE_DIR"
