# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# [tool.marimo.display]
# theme = "light"
# cell_output = "below"
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full", app_title="Playgrounds Demo")


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
    return


@app.cell
def _(mo):
    categories = sorted(
        [
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
        ]
    )

    frameworks = sorted(["sklearn", "tensorflow", "pytorch"])

    model_types = sorted(
        [
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
        ]
    )

    categories.append("other")
    frameworks.append("other")
    model_types.append("other")

    input_data_types = sorted(["Tabular", "Image", "Video", "Audio", "Text"])

    # **Model Card**
    mc_name = mo.ui.text(placeholder="The model card name.", full_width=True)
    mc_version = mo.ui.text(placeholder="The model card version.", full_width=True)
    mc_short_description = mo.ui.text(
        placeholder="500 characters max", max_length=500, full_width=True
    )
    mc_full_description = mo.ui.text_area()
    mc_author = mo.ui.text(placeholder="The model card author.", full_width=True)
    mc_keywords = mo.ui.text(
        placeholder="Keyword for the model card. Ex. 'demo-model, vision, icicle-ai'",
        full_width=True,
    )
    mc_input_data = mo.ui.text(
        placeholder="Url of the training dataset", kind="url", full_width=True
    )
    mc_input_type = mo.ui.dropdown(input_data_types)
    mc_output_data = mo.ui.text(
        placeholder="Model Repo Url", kind="url", full_width=True
    )
    mc_citation = mo.ui.text_area(
        placeholder="Optional: Citation or DOI of Published Work"
    )
    mc_foundational_model = mo.ui.text(
        placeholder="Optional: Foundation Patra Model Card ID", full_width=True
    )
    mc_category = mo.ui.dropdown(options=categories)
    mc_documentation = mo.ui.text(
        placeholder="Optional: Documentation URL", kind="url", full_width=True
    )

    # **AI Model**
    aim_name = mo.ui.text(placeholder="The AI model name.", full_width=True)
    aim_version = mo.ui.text(placeholder="The AI model version.", full_width=True)
    aim_description = mo.ui.text_area(
        placeholder="Description of the AI model.", full_width=True
    )
    aim_owner = mo.ui.text(placeholder="The AI model owner.", full_width=True)
    aim_location = mo.ui.text(
        placeholder="Downloadable url to model weights.", kind="url", full_width=True
    )
    aim_license = mo.ui.text(
        placeholder="Url license of the model", kind="url", full_width=True
    )
    aim_framework = mo.ui.dropdown(frameworks)
    aim_test_accuracy = mo.ui.text(
        placeholder="Test dataset accuracy.", full_width=True
    )
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
    return ai_model, model_card


@app.cell
def _(ai_model, mo, model_card):
    # Create state to store the patra_model_card instance
    # patra_model_card_state = mo.state(None)

    def create_model_card(data):
        import os
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
        patra_model_card = ModelCard(
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

        patra_model_card.validate()
        patra_model_card.save(f"/tmp/icicle-playgrounds/{model_card_data['Name']}.json")
        patra_model_card.submit(os.environ.get("PATRA_URL"))

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
    return patra_model_card_form


@app.cell
def _(mo, patra_model_card_form):
    import zipfile
    from io import BytesIO
    import httpx

    def publish_model(data):
        mc_name = patra_model_card_form.value.get("model_card")["Name"]

        def zip_files(files) -> bytes:
            """
            Zip multiple FileUploadResults into a single ZIP file.
            Returns the ZIP file as bytes.
            """
            # Create an in-memory bytes buffer
            zip_buffer = BytesIO()

            # Create a ZIP file in memory
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                with open(f"/tmp/icicle-playgrounds/{mc_name}.json", "r") as mc_file:
                    zip_file.writestr(f"{mc_name}.json", mc_file.read())

                    for file in files:
                        # Write each file's contents to the ZIP
                        zip_file.writestr(file.name, file.contents)

            # Get the bytes from the buffer
            zip_buffer.seek(0)
            return zip_buffer.getvalue()

        hf_token = data["hf_token"]
        repo_name = data["repo_name"]
        model_artifacts = zip_files(data["model_artifacts"])

        with httpx.Client(
            base_url="https://dev.develop.tapis.io/v3/mlhub/models-api"
        ) as client:
            client.post(
                "/artifacts",
                files={"file": (f"{mc_name}.zip", model_artifacts, "application/zip")},
            )
        print(hf_token)
        print(repo_name)
        print(model_artifacts)

        #     print(model_artifacts)
        # payload = {
        # "target_platform": "huggingface",
        # }

    # url = "https://dev.develop.tapis.io/v3/mlhub/models-api/artifacts"

    data = (
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
            hf_token=mo.ui.text(
                placeholder="HF Token...", kind="password", full_width=True
            ),
            repo_name=mo.ui.text(
                placeholder="ex: {account-name}/{repo-name}", full_width=True
            ),
            model_artifacts=mo.ui.file(multiple=True, max_size=1000000000),
        )
        .form(validate=publish_model)
    )
    return (data,)


@app.cell
def _(mo):
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
        async with agent:
            message = " ".join([m.content for m in messages])

            # # Set form data as environment variables for the agent to access
            # if publish_model_form.value:
            #     hf_token = publish_model_form.value.get("hf_token", "")
            #     repo_name = publish_model_form.value.get("repo_name", "")
            #     model_artifacts = publish_model_form.value.get("model_artifacts", [])

            #     print(hf_token)
            #     print(repo_name)
            #     print(model_artifacts)

            response = await agent.run(message)
            yield response.output

    chat = mo.ui.chat(
        icicle_playgrounds_agent,
        prompts=["Hello ICICLE, how are you?"],
        max_height=500,
        # allow_attachments=True,
    )
    return (chat,)


@app.cell
def _(chat, mo, patra_model_card_form, publish_model_form):
    sections = mo.accordion(
        {
            "Diagrams": mo.md(
                """
            ![diagram](https://raw.githubusercontent.com/ICICLE-ai/Playgrounds-Demo/refs/heads/main/PlaygroundsDemo.drawio.png)
            """
            ),
            "Create Patra Model Card": patra_model_card_form,
            "Publish Model Artifacts": publish_model_form,
        }
    )

    demo = mo.vstack(
        items=[
            mo.center(mo.md("# ICICLE Model & Data Playgrounds Demo")),
            sections,
            chat,
        ]
    )
    return (demo,)


@app.cell
def _(demo):
    demo
    return


@app.cell
def _(patra_model_card_form, publish_model_form):
    return


if __name__ == "__main__":
    app.run()
