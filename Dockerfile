FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv

RUN uv python install 3.13

# Create user early
RUN useradd -m icicle

WORKDIR /app

# Copy files as root first
RUN mkdir lib
COPY --link pyproject.toml .
COPY --link uv.lock .
COPY --link ./lib/playgrounds_agent-0.1.0-py3-none-any.whl ./lib/

# Set ownership of app directory and all contents
RUN chown -R icicle:icicle /app

# Switch to icicle user before creating venv
USER icicle

RUN uv sync --locked

RUN rm -rfd pyproject.toml uv.lock lib

COPY --link Playgrounds-Demo.py ./app.py

CMD [ "uv", "run", "marimo", "run", "app.py", "--host", "0.0.0.0", "-p", "5000"]
