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

singularity run \
    --bind "$HOME/singularity_data/postgres:/var/lib/postgresql/data" \
    postgres_16.sif \
    postgres \
        -c port=5433 \
        -c max_connections="$POSTGRES_MAX_CONNECTIONS" \
    > postgres.log 2>&1 &

POSTGRES_PID=$!

echo "PostgreSQL PID: $POSTGRES_PID"

echo "Starting pgAdmin..."

singularity run \
    --bind "$HOME/singularity_data/pgadmin:/var/lib/pgadmin" \
    pgadmin4.sif \
    > pgadmin.log 2>&1 &

PGADMIN_PID=$!

echo "pgAdmin PID: $PGADMIN_PID"

echo
echo "Services started:"
echo "PostgreSQL: localhost:5433"
echo "pgAdmin:    localhost:8080"
