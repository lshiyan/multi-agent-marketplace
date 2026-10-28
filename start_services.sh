
#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

POSTGRES_MAX_CONNECTIONS=${POSTGRES_MAX_CONNECTIONS:-100}

# Pass .env variables into Singularity.
while IFS='=' read -r var _; do
    [[ "$var" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || continue
    export "SINGULARITYENV_${var}=${!var-}"
done < .env

export SINGULARITYENV_PGADMIN_LISTEN_PORT=8080

POSTGRES_DATA="$HOME/singularity_data/postgres_data"
POSTGRES_SOCKET="$HOME/singularity_data/postgres_socket"
POSTGRES_IMAGE="postgres_16.sif"
POSTGRES_PORT=5433

mkdir -p "$POSTGRES_DATA" "$POSTGRES_SOCKET"

# Check whether PostgreSQL is accepting connections.
postgres_ready() {
    singularity exec "$POSTGRES_IMAGE" \
        pg_isready -h localhost -p "$POSTGRES_PORT" \
        >/dev/null 2>&1
}

if postgres_ready; then
    echo "PostgreSQL is already running on localhost:$POSTGRES_PORT."
    exit 0
fi

# Avoid starting another instance if the data directory
# is already owned by a PostgreSQL server.
if [[ -f "$POSTGRES_DATA/postmaster.pid" ]]; then
    echo "ERROR: PostgreSQL is not ready, but postmaster.pid exists."
    echo "Another instance may be starting or running."
    echo "Check existing PostgreSQL processes before retrying."
    exit 1
fi

echo "Starting PostgreSQL..."

singularity exec \
    --bind "$POSTGRES_DATA:/var/lib/postgresql/data" \
    --bind "$POSTGRES_SOCKET:/postgres_socket" \
    "$POSTGRES_IMAGE" \
    /usr/local/bin/docker-entrypoint.sh \
    postgres \
        -c port="$POSTGRES_PORT" \
        -c unix_socket_directories=/postgres_socket \
        -c max_connections="$POSTGRES_MAX_CONNECTIONS" \
    > postgres.log 2>&1 &

POSTGRES_PID=$!
echo "PostgreSQL launcher PID: $POSTGRES_PID"

# Wait up to 60 seconds for PostgreSQL to become ready.
for ((i = 0; i < 60; i++)); do
    if postgres_ready; then
        echo "PostgreSQL is ready on localhost:$POSTGRES_PORT."
        exit 0
    fi

    if ! kill -0 "$POSTGRES_PID" 2>/dev/null; then
        echo "ERROR: PostgreSQL process exited during startup."
        tail -30 postgres.log
        exit 1
    fi

    sleep 1
done

echo "ERROR: PostgreSQL did not become ready within 60 seconds."
tail -30 postgres.log
exit 1

