import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full", app_title="Playgrounds Demo", css_file="")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    from playgrounds_agent import agent

    async def icicle_playgrounds_agent(messages, config=None):
        async with agent:
            # response = await agent.run(messages[-1].content)
            message = " ".join([m.content for m in messages])
            response = await agent.run(message)
            yield response.output

    mo.ui.chat(
        icicle_playgrounds_agent,
        prompts=["Hello ICICLE, how are you?"],
        max_height=5000,
        # show_configuration_controls=True,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
