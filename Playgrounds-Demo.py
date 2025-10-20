import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full", app_title="Playgrounds Demo", css_file="")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.sidebar(
        [
            mo.md("# ICICLE AI: Model & Data Playgrounds"),
            mo.nav_menu(
                {
                    "https://icicle.ai": f"{mo.icon('lucide:home')} Home",
                    "Links": {
                        "https://github.com/icicle-ai": "GitHub",
                        "https://huggingface.co/icicle-ai": "Hugging Face",
                    },
                },
                orientation="vertical",
            ),
        ],
    )


@app.cell
def _(mo):
    categories = sorted([
        "classification",
        "regression",
        "clustering",
        "anomaly detection",
        "dimensionality reduction",
        "reinforcement learning",
        "natural language processing",
        "computer vision",
        "recommendation systems",
        "time series forecasting",
        "graph learning",
        "graph neural networks",
        "generative modeling",
        "transfer learning",
        "self-supervised learning",
        "semi-supervised learning",
        "unsupervised learning",
        "causal inference",
        "multi-task learning",
        "metric learning",
        "density estimation",
        "multi-label classification",
        "ranking",
        "structured prediction",
        "neural architecture search",
        "sequence modeling",
        "embedding learning",
    ])

    frameworks = sorted(["sklearn", "tensorflow", "pytorch"])

    model_types = sorted([
        "cnn",
        "decision_tree",
        "dnn",
        "rnn",
        "svm",
        "kmeans",
        "llm",
        "random_forest",
        "lstm",
        "gnn",
    ])

    categories.append("other")
    frameworks.append("other")
    model_types.append("other")

    input_data_types = sorted(["Tabular", "Image", "Video", "Audio", "Text"])

    # **Model Card**
    mc_name = mo.ui.text(placeholder="The model card name.")
    mc_version = mo.ui.text(placeholder="The model card version.")
    mc_short_description = mo.ui.text(placeholder="500 characters max", max_length=500)
    mc_full_description = mo.ui.text_area()
    mc_author = mo.ui.text(placeholder="The model card author.")
    mc_keywords = mo.ui.text(
        placeholder="Keyword for the model card. Ex. 'demo-model, vision, icicle-ai'"
    )
    mc_input_data = mo.ui.text(placeholder="Url of the training dataset", kind="url")
    mc_input_type = mo.ui.dropdown(input_data_types)
    mc_output_data = mo.ui.text(placeholder="Model Repo Url", kind="url")
    mc_citation = mo.ui.text_area(
        placeholder="Optional: Citation or DOI of Published Work"
    )
    mc_foundational_model = mo.ui.text(
        placeholder="Optional: Foundation Patra Model Card ID"
    )
    mc_category = mo.ui.dropdown(options=categories)
    mc_documentation = mo.ui.text(placeholder="Optional: Documentation URL", kind="url")

    # **AI Model**
    aim_name = mo.ui.text(placeholder="The AI model name.")
    aim_version = mo.ui.text(placeholder="The AI model version.")
    aim_description = mo.ui.text_area(placeholder="Description of the AI model.")
    aim_owner = mo.ui.text(placeholder="The AI model owner.")
    aim_location = mo.ui.text(
        placeholder="Downloadable url to model weights.", kind="url"
    )
    aim_license = mo.ui.text(placeholder="Url license of the model", kind="url")
    aim_framework = mo.ui.dropdown(frameworks)
    aim_test_accuracy = mo.ui.text(placeholder="Test dataset accuracy.")
    aim_type = mo.ui.dropdown(model_types)
    aim_structure = mo.ui.text_area(placeholder="Optional: Model structure.")
    aim_metrics = mo.ui.text_area(
        placeholder="Optional: Metrics Ex. {Test Accuracy: 0.9}, {Test Loss: 0.001}"
    )
    aim_labels = mo.ui.text_area(
        placeholder="Optional: Inference labels. Ex. {0: dog, 1: cat}"
    )

    ai_model = mo.ui.dictionary(
        {
            "Name": aim_name,
            "Version": aim_version,
            "Description": aim_description,
            "Owner": aim_owner,
            "Location": aim_location,
            "License": aim_license,
            # "Structure": aim_structure,
            "Framework": aim_framework,
            "Type": aim_type,
            "Test Accuracy": aim_test_accuracy,
            # "Metrics": aim_metrics,
            # "Inference Labels": aim_labels,
        },
        label="AI Model",
    )

    model_card = mo.ui.dictionary(
        {
            "Name": mc_name,
            "Version": mc_version,
            "Short Description": mc_short_description,
            "Full Description": mc_full_description,
            "Keywords": mc_keywords,
            "Author": mc_author,
            "Input Type": mc_input_type,
            # "Input Data": mc_input_data,
            # "Output Data": mc_output_data,
            # "Citation": mc_citation,
            # "Foundational Model": mc_foundational_model,
            "Category": mc_category,
            # "Documentation": mc_documentation,
        },
        label="Model Card",
    )

    return (model_card, ai_model)


@app.cell
def _(ai_model, mo, model_card):

    patra_model_card = mo.vstack(
        [
            mo.md("## Model Card Information"),
            model_card,
            mo.md("## AI Model Information"),
            ai_model,
        ]
    )

    def create_model_card(data):
        from patra_toolkit import ModelCard, AIModel
        ai_model_data = data["ai_model"]
        model_card_data = data["model_card"]
        ai_model = AIModel(
            name=ai_model_data["Name"],
            version=ai_model_data["Version"],
            description=ai_model_data["Description"],
            owner=ai_model_data["Owner"],
            location=ai_model_data["Location"],
            license=ai_model_data["License"],
            framework=ai_model_data["Framework"][0],
            model_type=ai_model_data["Type"][0],
            test_accuracy=float(ai_model_data["Test Accuracy"]),
        )
        model_card = ModelCard(
            name=model_card_data["Name"],
            version=model_card_data["Version"],
            short_description=model_card_data["Short Description"],
            full_description=model_card_data["Full Description"],
            keywords=model_card_data["Keywords"],
            author=model_card_data["Author"],
            input_type=model_card_data["Input Type"][0],
            category=model_card_data["Category"][0],
            ai_model=ai_model,
        )
        print(model_card)

        model_card.validate()
        # model_card.submit()
        print(data["model_card"])

    patra_model_card_form = (
        mo.md(
            """
            **Patra Model Card**
            ## Model Card Information
            {model_card}

            ## AI Model Information
            {ai_model}
            """
        )
        .batch(model_card=model_card, ai_model=ai_model)
        .form(
            # submit_button_label="Validate",
            # submit_button_tooltip="Validate your Patra Model Card information.",
            show_clear_button=True,
            validate=create_model_card,
        )
    )

    return (patra_model_card_form,)


@app.cell
def _(mo):
    input_form = (
        mo.md(
            """
            **Hugging Face Hub Token**

            {hf_token}

            **Model Repo Name**

            {repo_name}

            **Model Artifacts**
            {model_artifacts}
            """
        )
        .batch(
            hf_token=mo.ui.text(placeholder="HF Token...", kind="password"),
            repo_name=mo.ui.text(placeholder="ex: {account-name}/{repo-name}"),
            model_artifacts=mo.ui.file(multiple=True),
        )
        .form()
    )
    return (input_form,)


@app.cell
def _(input_form, mo):
    import xml.etree.ElementTree as ET
    from pathlib import Path

    from playgrounds_agent import build_agent

    xml_file = Path(__file__).parent / "prompts" / "instructions.xml"

    tree = ET.parse(xml_file)
    root = tree.getroot()

    inst_v2 = root.find(".//version/v2/instructions")
    if inst_v2 is None:
        raise FileNotFoundError

    # Get all text content from the element, including text with proper formatting
    instructions_text = "".join(inst_v2.itertext()).strip()

    agent = build_agent(instructions=instructions_text)

    async def icicle_playgrounds_agent(messages, config=None):
        try:
            async with agent:
                message = " ".join([m.content for m in messages])

                # Set form data as environment variables for the agent to access
                if input_form.value:
                    hf_token = input_form.value.get("hf_token", "")
                    repo_name = input_form.value.get("repo_name", "")
                    model_artifacts = input_form.value.get("model_artifacts", [])

                    print(hf_token)
                    print(repo_name)
                    print(model_artifacts)

                response = await agent.run(message)
                yield response.output
        except ExceptionGroup as eg:
            for exc in eg.exceptions:
                yield f"⚠️ MCP Server Error: {type(exc).__name__}: {exc}"
        except Exception as e:
            yield f"⚠️ Error: {type(e).__name__}: {e}"

    chat = mo.ui.chat(
        icicle_playgrounds_agent,
        prompts=["Hello ICICLE, how are you?"],
        max_height=500,
        # allow_attachments=True,
    )
    return (chat,)


@app.cell
def _(chat, input_form, mo, patra_model_card_form):
    sections = mo.accordion(
        {
            "1. Introduction": mo.md(
                """
            # Introduction
            ![diagram](./PlaygroundsDemo.drawio.png)
            """
            ),
            "2. Create Patra Model Card": patra_model_card_form,
            "3. Publish Model Artifacts": input_form,
        }
    )

    demo = mo.vstack(items=[sections, chat])
    return (demo,)


@app.cell
def _(demo):
    demo
    return


if __name__ == "__main__":
    app.run()
