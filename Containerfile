FROM quay.io/sclorg/python-312-c10s:latest as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_NATIVE_TLS=true \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/opt/app-root/src/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/app-root \
    UV_NO_CACHE=1

USER 1001

WORKDIR /opt/app-root/src

# Install project
COPY --chown=1001:0 . .
RUN uv sync --no-dev

CMD [ "/bin/bash", "./app.sh" ]
