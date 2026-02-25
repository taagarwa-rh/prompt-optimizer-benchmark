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
