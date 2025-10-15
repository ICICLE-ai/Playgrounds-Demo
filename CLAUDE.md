# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Marimo-based interactive notebook application that integrates with ICICLE Playgrounds and uses a custom agent system. The application provides a chat interface for interacting with an AI agent.

## Development Setup

### Environment Management

This project uses `uv` for dependency management. The virtual environment is located in `.venv/`.

To install dependencies:
```bash
uv sync
```

### Running the Application

The main application is a Marimo notebook. To run it:
```bash
marimo edit "Playgrounds Demo.py"
```

Or to run in production mode:
```bash
marimo run "Playgrounds Demo.py"
```

### Testing

No test framework is currently configured. If adding tests, use pytest:
```bash
uv add --dev pytest
uv run pytest
```

## Architecture

### Core Components

**Marimo Application** (`Playgrounds Demo.py`)
- Main entry point is a Marimo reactive notebook
- Creates an interactive chat interface using `mo.ui.chat`
- Integrates with the playgrounds_agent via async wrapper function

**Playgrounds Agent**
- Custom agent provided as a wheel file in `lib/playgrounds_agent-0.1.0-py3-none-any.whl`
- Imported as `from playgrounds_agent import agent`
- Agent runs asynchronously and processes chat messages
- Messages are concatenated from the conversation history before being sent to the agent

### Agent Integration Pattern

The chat interface uses an async generator pattern:
```python
async def icicle_playgrounds_agent(messages, config=None):
    async with agent:
        message = " ".join([m.content for m in messages])
        response = await agent.run(message)
        yield response.output
```

This concatenates all previous messages into a single string before processing.

### Environment Configuration

The `.env` file contains critical configuration:
- `MODEL_URL`: Endpoint for the LLM model (default: `http://localhost:8084/v1`)
- `API_KEY`: OpenAI API key for model access
- `MODEL_NAME`: Model to use (currently `gpt-4o-mini`)
- `PLAYGROUNDS_MCP`: MCP server endpoint (default: `http://localhost:3000/mcp`)
- `DOCKER_HOST`: Optional Docker socket configuration for Colima

### Dependencies

Key dependencies include:
- **marimo**: Interactive notebook framework
- **icicle-playgrounds**: ICICLE Playgrounds integration
- **pydantic-ai**: AI agent framework
- **patra-toolkit**: Additional toolkit
- **transformers, torch, torchvision**: ML/DL frameworks
- **datasets, evaluate**: ML dataset and evaluation tools

The custom `playgrounds-agent` package is installed from a local wheel file.

## Code Style

Use `ruff` for linting and formatting:
```bash
uv run ruff check .
uv run ruff format .
```
