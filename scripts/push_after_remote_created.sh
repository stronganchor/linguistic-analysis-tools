#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  cat >&2 <<'EOF'
Usage:
  scripts/push_after_remote_created.sh https://github.com/OWNER/REPO.git

Run this after creating an empty GitHub repository.
EOF
  exit 1
fi

remote_url="$1"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$remote_url"
else
  git remote add origin "$remote_url"
fi

git push -u origin main
