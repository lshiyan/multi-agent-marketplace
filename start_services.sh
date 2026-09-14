#!/bin/bash

set -a
source .env
set +a

POSTGRES_MAX_CONNECTIONS=${POSTGRES_MAX_CONNECTIONS:-100}

# Pass .env variables into Singularity containers
for var in $(cut -d= -f1 .env | grep -v '^#' | grep -v '^$'); do
    export "SINGULARITYENV_${var}=${!var}"
done

# pgAdmin normally listens on port 80 inside Docker.
# With Singularity there is no Docker-style port mapping,
# so make pgAdmin listen directly on 8080.
export SINGULARITYENV_PGADMIN_LISTEN_PORT=8080

mkdir -p "$HOME/singularity_data/postgres"
mkdir -p "$HOME/singularity_data/pgadmin"

echo "Starting PostgreSQL..."

singularity exec \
  --bind "$HOME/singularity_data/postgres_data:/var/lib/postgresql/data" \
  --bind "$HOME/singularity_data/postgres_socket:/postgres_socket" \
  postgres_16.sif \
  /usr/local/bin/docker-entrypoint.sh \
  postgres \
    -c port=5433 \
    -c unix_socket_directories=/postgres_socket \
    -c max_connections="${POSTGRES_MAX_CONNECTIONS:-100}"

POSTGRES_PID=$!

echo "PostgreSQL PID: $POSTGRES_PID"

echo
echo "Services started:"
echo "PostgreSQL: localhost:5433"
