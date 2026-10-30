#!/usr/bin/env bash
set -euo pipefail

POSTGRES_DATA="$HOME/singularity_data/postgres_data"
POSTGRES_SOCKET="$HOME/singularity_data/postgres_socket"
POSTGRES_IMAGE="postgres_16.sif"
POSTGRES_PORT=5433

if [[ ! -f "$POSTGRES_DATA/postmaster.pid" ]]; then
    echo "No PostgreSQL instance found in $POSTGRES_DATA (no postmaster.pid)."
    exit 0
fi

if [[ ! -f "$POSTGRES_IMAGE" ]]; then
    echo "ERROR: Singularity image not found: $POSTGRES_IMAGE" >&2
    exit 1
fi

# pg_ctl must run as the same OS user that owns the PostgreSQL server.
# Bind the same directories as the startup script.
echo "Stopping PostgreSQL using data directory: $POSTGRES_DATA"
singularity exec \
    --bind "$POSTGRES_DATA:/var/lib/postgresql/data" \
    --bind "$POSTGRES_SOCKET:/postgres_socket" \
    "$POSTGRES_IMAGE" \
    pg_ctl -D /var/lib/postgresql/data -m fast -w -t 60 stop

echo "PostgreSQL stopped."
