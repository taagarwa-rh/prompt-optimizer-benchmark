_default:
    @ just --list

# Run recipes for MR approval
pre-mr: format lint

# Formats Code
format:
    uv run ruff check --select I --fix .
    uv run ruff format .

# Lints Code
lint *options:
    uv run ruff check . {{ options }}

build:
    podman build -t prompt-optimization-benchmark:latest -f Containerfile

test-container: build
    podman run --rm \
        -v ./results:/opt/app-root/src/results \
        -it prompt-optimization-benchmark:latest \
        /bin/bash