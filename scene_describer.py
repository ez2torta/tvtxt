import os
import time
from uuid import uuid4

import modal


GPU_TYPE = os.environ.get("GPU_TYPE", "l40s")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
MINUTES = 60  # seconds
MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

# We download it from the Hugging Face Hub using the Python function below.
def download_model_to_image():
    import transformers
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],
    )

    # otherwise, this happens on first inference
    transformers.utils.move_cache()


# Modal runs Python functions on containers in the cloud.
# The environment those functions run in is defined by the container's `Image`.
# The block of code below defines our example's `Image`.
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "transformers==4.47.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "requests==2.32.3",
        "starlette==0.41.2",
        "torch==2.4.0",
        "outlines",
        "datasets",
        "sentencepiece",
        "accelerate>=0.26.0",
        "pillow",
        "rich"
    )
    .run_function(  # download the model by running a Python function
        download_model_to_image
    )
)

app = modal.App("tvtxt-vlm")

@app.cls(
    gpu=GPU_CONFIG,
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,
    image=vlm_image,
)
@modal.concurrent(max_inputs=100)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    def start_runtime(self):
        import outlines
        import torch
        from transformers import Qwen2VLForConditionalGeneration

        # Create the Outlines model
        self.model = outlines.models.transformers_vision(
            MODEL_PATH,
            model_class=Qwen2VLForConditionalGeneration,
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            },
            processor_kwargs={
                "device": "cuda", # set to "cpu" if you don't have a GPU
            },
        )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict) -> dict:
        import requests
        from pydantic import BaseModel, Field
        from io import BytesIO
        from urllib.request import urlopen
        from transformers import AutoProcessor
        import outlines

        from PIL import Image

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_url = request.get("image_url")
        if image_url is None:
            image_url = (
                "https://alonsoastroza.com/projects/ft-hackathon/avello.jpg"
            )

        response = requests.get(image_url)
        response.raise_for_status()

        # Define the schema using Pydantic
        class SceneMovieScript(BaseModel):
            heading: str = Field(..., description="One line description of the location and time of the day. Be concise. Example: EXT SUBURBAN HOME - NIGHT")
            action: str = Field(..., description="The description of the scene. Be very concise and clear.")
            character: str = Field(..., description="The name of the character speaking. If there is no character, use 'Narrator'.")

        def get_image_from_url(image_url):
            img_byte_stream = BytesIO(urlopen(image_url).read())
            return Image.open(img_byte_stream).convert("RGB")

        # Set up the content you want to send to the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        # The image is provided as a PIL Image object
                        "type": "image",
                        "image": get_image_from_url(image_url),
                    },
                    {
                        "type": "text",
                        "text": f"""You are an expert at writing movie scripts based on images.
                        Please describe the scene in a movie script format. Be concise and clear.

                        Return the information in the following JSON schema:
                        {SceneMovieScript.model_json_schema()}
                    """},
                ],
            }
        ]

        # Convert the messages to the final prompt
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        script_generator = outlines.generate.json(
            self.model,
            SceneMovieScript,

        )

        # Generate the receipt summary
        result = script_generator(prompt, [get_image_from_url(image_url)])

        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )
        print(f"Response: {result}")
    
        return {"heading": result.heading, "action": result.action, "character": result.character}

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        self.runtime.shutdown()