# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# [tool.marimo.display]
# theme = "light"
# cell_output = "below"
# ///

import marimo
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to console
    ],
)

logger = logging.getLogger(__name__)

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
                    "https://icicle-ai.github.io/training-catalog": f"{mo.icon('lucide:doc')} Training Catalog",
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
            # Create & Submit a Patra Model Card to the Knowledge Graph (KG)
            The Patra Knowledge Base is a system designed to manage and track 
            AI/ML models, with the objective of making them more accountable 
            and trustworthy. It's a key part of the Patra ModelCards framework,
            which aims to improve transparency and accountability in AI/ML 
            models throughout their entire lifecycle. This includes the model's 
            initial training phase, subsequent deployments, and ongoing usage, 
            whether by the same or different individual.

            Patra Model Cards are detailed records that provide essential 
            information about each AI/ML model. This information includes 
            technical details like the model's accuracy and latency, but it 
            goes beyond that to include non-technical aspects such as 
            fairness, explainability, and the model's behavior in various 
            deployment environments. This holistic approach is intended to 
            create a comprehensive understanding of the model's strengths 
            and weaknesses, enabling more informed decisions about its use 
            and deployment
            
            For more information visit the [ICICLE AI Training Catalog](https://icicle-ai.github.io/training-catalog/docs/category/patra-kg-2).

            ---
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

    from pydantic import BaseModel

    class MLHubArtifactPayload(BaseModel):
        name: str
        model_type: str
        version: str
        framework: str
        license: str

    def publish_model(data):
        if not patra_model_card_form.value:
            raise ValueError("Please create a Patra Model Card first")

        model_card = patra_model_card_form.value.get("model_card")
        ai_model = patra_model_card_form.value.get("ai_model")

        if not model_card or not ai_model:
            raise ValueError("Model card or AI model data is missing")

        mc_name = model_card["Name"]

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
            response = client.post(
                "/artifacts",
                files={"file": (f"{mc_name}.zip", model_artifacts, "application/zip")},
            )

            response.raise_for_status()
            print(response.json())
            artifact_id = response.json()["result"]

            response = client.post(
                f"/artifacts/{artifact_id}",
                json=MLHubArtifactPayload(
                    name=repo_name,
                    model_type=ai_model["Type"],
                    version=ai_model["Version"],
                    framework=ai_model["Framework"],
                    license=ai_model["License"],
                ).model_dump(),
            )

            response.raise_for_status()
            print(response.json())

            response = client.post(
                f"/artifacts/{artifact_id}/publications",
                json={"target_platform": "huggingface"},
                headers={"Authorization": f"Bearer {hf_token}"},
            )
            print(response.json())

    publish_model_form = (
        mo.md(
            """
        # Publish Model Artifacts with MLHub to Hugging Face Hub
        You can use MLHub to publish your model's artifacts to Hugging Face. 
        To do so you first need to create a zip archive of the artifacts. 
        You can either create an archive using your file browser or you can use the 
        follwing snippets to do so in your Python script.
        ```python
        import zipfile
        from io import BytesIO

        def zip_files(filenames: list[str]) -> bytes:
            # Create an in-memory bytes buffer.
            zip_buffer = BytesIO()

            # Create a ZIP file in memory
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip:
                for filename in filenames:
                    # Archives file from filesystem
                    zip.write(filename)
            # Set file pointer to the begining of the zip archive and return the bytes value.
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
        ```

        With our artifacts archived in a zip file. We can start using MLHub to 
        publish these to our Hugging Face Hub Model Repo.

        MLHub requires us to do this in folowing order,
        1.  Upload our archive to MLHub,
        ```python
        import httpx
        from pydantic_settings import BaseSettings

        class PublishingParams(BaseSettings):
            model_config = SettingsConfigDict(env_file="*.env*", env_file_encoding="utf-8")

            hf_token: str
            tapis_token: str
            model_repo: str

        params = PublishingParams()

        my_model_artifacts = [
            "patra_model_card.json", # Patra Model Card we created
            "MyModel.pt", # Saved PyTorch trained weights,
            "README.md", # Repo README,
            "MyModel.py", # MyModel python script
        ]

        base_url = "https://dev.develop.tapis.io/v3/mlhub/model-api"
        response = httpx.post(
            url=f"{{ base_url }}/artifacts",
            files={{ "file": ("MyModel.zip", zip_files(my_model_artifacts), "application/zip") }}
        )

        # Raises an exception if it is not a successful response.
        response.raise_for_status()

        # If successful, MLHub will return the created UUID for the artifact.
        artifact_uuid = response.json()["result"]

        ```
        2. Create Model Artifact Metadata.
        - This meta is used in order for MLHub to provision and deploy your model using a Tapis sytem.
        ```python
        from pydantic import BaseModel
        class MLHubArtifactMetadata(Basemodel):
            name: str # Platform Repo Name (ex. for HuggingFace its {{ account }}/{{ repo }}) REQUIRED FOR MLHUB TO PROBLISH TO DESIRED PLATFORM
            model_type: str # The model type (cnn, transformer, llm, etc)
            version: str # The version of the model
            framework: str # Framework used to create the model. (PyTorch, Tensorflow, etc)
            license: str # Hyperlink to the license to use the model.

        metadata = MLHubArtifactMetadata(
            name=params.model_repo, # Taken from the PublishParams
            model_type="CNN",
            version="0.1",
            framework="PyTorch",
            license="https://huggingface.co/icicle-ai/MyModel/blob/main/LICENSE"
        )

        reponse = Client.post(
            url=f"{{ base_url }}/artifacts/{{ artifact_id }}",
            json=metadata.model_dump()
        )

        # Raise to see if any issues encountered. 
        response.raise_for_status()
        ```

        3. With the artifacts and their metadata successfully create and 
        ingested into MLHub, we can use MLHub to publish it to our desired platform.
        ```python
        # Publishing to Hugging Face Hub
        
        reponse = httpx.post(
            url=f"{{ base_url }}/artifacts/{{ artifact_id }}/publications",
            json={{ "target_platform": "huggingface" }},
            headers={{ "Authorization": f"Bearer {{ config.hf_token }}" }}
        )
        
        # Raise to see if any issues encountered. 
        response.raise_for_status()
        ```
        ---
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

    return (publish_model_form,)


@app.cell
def _(mo):
    import xml.etree.ElementTree as ET
    from pathlib import Path

    from playgrounds_agent import build_agent

    xml_file = Path(__file__).parent / "prompts" / "instructions.xml"

    tree = ET.parse(xml_file)
    root = tree.getroot()

    inst = root.find(".//version/v3/instructions")
    if inst is None:
        raise FileNotFoundError

    # Get all text content from the element, including text with proper formatting
    instructions_text = "".join(inst.itertext()).strip()
    print(instructions_text)
    agent = build_agent(instructions=instructions_text, max_retries=3)

    async def icicle_playgrounds_agent(messages, config=None):
        async with agent:
            message = " ".join([m.content for m in messages])

            # # Set form data as environment variables for the agent to access
            # if publish_model_form.value:
            #     hf_token = publish_model_form.value.get("hf_token", "")
            #     repo_name = publish_model_form.value.get("riepo_name", "")
            #     model_artifacts = publish_model_form.value.get("model_artifacts", [])

            #     print(hf_token)
            #     print(repo_name)
            #     print(model_artifacts)

            response = await agent.run(message)
            yield response.output

    chat = mo.ui.chat(
        icicle_playgrounds_agent,
        prompts=[
            "Hello ICICLE, how are you?",
            "My name is Carlos and I am an an Animal Ecologist. I am interested in using AI to help me with my research.",
            "I am currently studying animal behaviors through the use of camera trap images that I have deployed.",
        ],
        max_height=500,
        # allow_attachments=True,
    )
    return (chat,)


@app.cell
def _(chat, mo, patra_model_card_form, publish_model_form):
    sections = mo.accordion(
        {
            "Overview": mo.md(
                """
            The ICICLE Model & Data Playgrounds is a platform that allows you to discover and plug-n-play with AI models. 
            This is all possible through the use of,
            - Patra Knowledge Graph and Model Cards to collect and store model metadata
            in order to aide the discovery of models. 
            - The Playgrounds Agent and MLHub to discover models from multple paltform including Patra, Hugging Face, and GitHub.
            - MLHub and Patra to publish models to your desired platform.
            - Tapis Workflows to Plug-n-Play with these models in a containerized and safe platform.

            
            ![diagram](https://raw.githubusercontent.com/ICICLE-ai/Playgrounds-Demo/refs/heads/main/static/ICICLE Model & Data Playgrounds.drawio.png)
            """
            ),
            "Create Patra Model Card": patra_model_card_form,
            "Publish Model Artifacts": publish_model_form,
        }
    )

    demo = mo.vstack(
        items=[
            mo.center(mo.md(
                """
                #ICICLE Model & Data Playgrounds: MLHub & Patra Demo

                """
            )),
            sections,
            chat,
        ]
    )
    return (demo,)


@app.cell
def _(demo):
    demo
    return


if __name__ == "__main__":
    app.run()
