# Pydantic-AI Dependency Injection Guide

This document explains how to use dependency injection with pydantic-ai to pass form data (like Hugging Face tokens, repo names, and uploaded files) to your agent.

## Overview

Pydantic-AI's dependency injection allows you to pass data that tools and the agent can access in a type-safe way. It's very similar to FastAPI's dependency injection system.

## How It Works

### 1. Define a Dependency Class

First, create a class (usually a dataclass or Pydantic model) that holds the data you want to pass:

```python
from dataclasses import dataclass

@dataclass
class FormData:
    hf_token: str | None = None
    repo_name: str | None = None
    model_artifacts: list | None = None  # List of file objects with .name and .contents
```

### 2. Type the Agent with the Dependency

When creating the agent, you specify what type of dependencies it accepts:

```python
from pydantic_ai import Agent

# This tells the agent it can receive FormData as dependencies
agent = Agent[None, FormData](  # [RunContext, Dependencies]
    model=model,
    instructions=instructions,
    toolsets=[mlhub_mcp],
)
```

The generic parameters are:
- **First type** (`None`): The run context type (we don't need it for basic usage)
- **Second type** (`FormData`): The dependency type

### 3. Pass Dependencies at Runtime

When you run the agent, you pass an instance of your dependency class:

```python
# Create the dependency instance with actual data
form_data = FormData(
    hf_token="hf_abc123",
    repo_name="user/model-name",
    model_artifacts=[file1, file2]  # Actual file objects from Marimo form
)

# Pass it when running the agent
response = await agent.run(message, deps=form_data)
```

### 4. Access Dependencies in Tools

Any tools can access the dependency data through the `RunContext`:

```python
from pydantic_ai import RunContext

@agent.tool
async def upload_to_huggingface(ctx: RunContext[FormData], model_file: str) -> str:
    """Upload a model file to Hugging Face"""

    # Access the form data through ctx.deps
    token = ctx.deps.hf_token
    repo = ctx.deps.repo_name
    files = ctx.deps.model_artifacts

    if not token or not repo:
        return "Missing HF token or repo name"

    # Use the data - save files and upload
    for file_obj in files:
        # file_obj.name - filename
        # file_obj.contents - file bytes as bytes
        path = f"/tmp/{file_obj.name}"
        with open(path, 'wb') as f:
            f.write(file_obj.contents)

        # Upload to HuggingFace
        hf_api.upload_file(
            token=token,
            repo_id=repo,
            path_or_fileobj=path,
            path_in_repo=file_obj.name
        )

    return f"Uploaded {len(files)} files to {repo}"
```

## Benefits Over Environment Variables

1. **Type Safety**: The type system ensures you're accessing the right data with proper types
2. **Scoped**: Dependencies are scoped to each agent run, not global
3. **Testable**: Easy to test with different dependency values
4. **Clean**: Tools explicitly declare what they need via type hints
5. **Documentation**: The types serve as documentation
6. **No Side Effects**: No global state modification
7. **Memory Efficient**: Files stay in memory until actually needed, no unnecessary disk I/O

## Implementation for This Project

### Step 1: Modify `playgrounds_agent/agent.py`

Update the `build_agent` function to accept a dependency type:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from typing import TypeVar

from playgrounds_agent.config import config
from playgrounds_agent.model import build_model
from playgrounds_agent.providers import get_provider

# Create a generic type variable for dependencies
DepsT = TypeVar('DepsT')

def build_agent(instructions: str, deps_type: type[DepsT] | None = None) -> Agent:
    try:
        mlhub_mcp = MCPServerStreamableHTTP(url=config.mlhub_mcp)
        provider = get_provider(config.model_provider, api_key=config.api_key)
        model = build_model(config.model_name, provider=provider)

        if deps_type:
            # Create agent with dependency type
            agent = Agent[None, DepsT](
                model=model,
                instructions=instructions,
                toolsets=[mlhub_mcp],
            )
        else:
            # Create agent without dependencies (backward compatible)
            agent = Agent(
                model=model,
                instructions=instructions,
                toolsets=[mlhub_mcp],
            )

        return agent
    except Exception as e:
        print(e)
        raise e
```

### Step 2: Update Marimo Notebook

In your `Playgrounds-Demo.py` notebook:

```python
from dataclasses import dataclass

@dataclass
class FormData:
    hf_token: str | None = None
    repo_name: str | None = None
    model_artifacts: list | None = None

# Build agent with FormData as dependency type
agent = build_agent(instructions=instructions_text, deps_type=FormData)

async def icicle_playgrounds_agent(messages, config=None):
    try:
        async with agent:
            message = " ".join([m.content for m in messages])

            # Create FormData from input_form
            form_data = FormData()
            if input_form.value:
                form_data.hf_token = input_form.value.get('hf_token', '')
                form_data.repo_name = input_form.value.get('repo_name', '')
                form_data.model_artifacts = input_form.value.get('model_artifacts', [])

            # Pass as deps - now available to all tools!
            response = await agent.run(message, deps=form_data)
            yield response.output
    except Exception as e:
        yield f"⚠️ Error: {e}"
```

## Comparison with FastAPI

If you're familiar with FastAPI, the patterns are very similar:

### FastAPI:
```python
from fastapi import Depends

def get_db():
    return Database()

@app.get("/users/{user_id}")
async def read_user(user_id: int, db: Database = Depends(get_db)):
    # db is automatically injected
    return {"user": db.query(user_id)}
```

### Pydantic-AI:
```python
from pydantic_ai import RunContext

agent = Agent[None, Database](...)

@agent.tool
async def get_user(ctx: RunContext[Database], user_id: int):
    # ctx.deps is automatically injected
    return ctx.deps.query(user_id)

# When running:
await agent.run("get user 123", deps=Database())
```

## Key Parallels

1. **Type hints drive injection**: Both systems use type annotations to know what to inject
2. **Scoped per request/run**:
   - FastAPI: Dependencies are scoped per HTTP request
   - Pydantic-AI: Dependencies are scoped per `agent.run()` call
3. **Automatic passing**: You don't manually pass deps everywhere - the framework does it
4. **Testability**: Easy to mock dependencies in tests

## How Files Flow Through the System

1. User uploads files via Marimo form → `input_form.value.get('model_artifacts')`
2. Files are packaged into `FormData` instance with `.name` and `.contents` attributes
3. `FormData` is passed to `agent.run(message, deps=form_data)`
4. Agent tools access files via `ctx.deps.model_artifacts`
5. Tools can iterate over files, save them, upload them, etc.
6. Files stay in memory until a tool explicitly saves them to disk

## Example: Complete Tool Implementation

```python
from pydantic_ai import RunContext
from huggingface_hub import HfApi

@agent.tool
async def upload_model_artifacts(ctx: RunContext[FormData]) -> str:
    """
    Upload model artifacts to Hugging Face Hub.
    Requires HF token and repo name from the form.
    """
    token = ctx.deps.hf_token
    repo = ctx.deps.repo_name
    files = ctx.deps.model_artifacts

    # Validation
    if not token:
        return "Error: Hugging Face token not provided"
    if not repo:
        return "Error: Repository name not provided"
    if not files:
        return "Error: No model artifacts uploaded"

    # Initialize HF API
    api = HfApi()

    uploaded_files = []
    for file_obj in files:
        # Save file temporarily
        temp_path = f"/tmp/{file_obj.name}"
        with open(temp_path, 'wb') as f:
            f.write(file_obj.contents)

        # Upload to HuggingFace
        try:
            api.upload_file(
                token=token,
                repo_id=repo,
                path_or_fileobj=temp_path,
                path_in_repo=file_obj.name,
                repo_type="model"
            )
            uploaded_files.append(file_obj.name)
        except Exception as e:
            return f"Error uploading {file_obj.name}: {e}"

    return f"Successfully uploaded {len(uploaded_files)} files to {repo}: {', '.join(uploaded_files)}"
```

## Testing with Different Dependencies

```python
# Test with mock data
test_form_data = FormData(
    hf_token="test_token",
    repo_name="test/repo",
    model_artifacts=[]
)

response = await agent.run("test message", deps=test_form_data)
```

## Summary

Dependency injection in pydantic-ai provides a clean, type-safe way to pass context (like form data, uploaded files, API tokens) to your agent and its tools. It's similar to FastAPI's system and offers significant advantages over global state or environment variables.
