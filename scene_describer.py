import os
import time
from uuid import uuid4
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
import torch
import outlines
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv
import requests
from base64 import b64decode

# Cargar variables de entorno desde .env si existe
load_dotenv()

# MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2-VL-7B-Instruct")
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" # probando ahora que tan veloz puede ser
# MODEL_PATH = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
HF_TOKEN = os.environ.get("HF_TOKEN")

def get_image_from_url(image_url):
    img_byte_stream = BytesIO(urlopen(image_url).read())
    return Image.open(img_byte_stream).convert("RGB")

class SceneMovieScript(BaseModel):
    heading: str = Field(..., description="One line description of the location and time of the day. Be concise. Example: EXT SUBURBAN HOME - NIGHT")
    action: str = Field(..., description="The description of the scene. Be very concise and clear.")
    character: str = Field(..., description="The name of the character speaking. If there is no character, use 'Narrator'.")

# Inicialización del modelo y procesador al inicio
print("Cargando modelo y procesador...")

# Si hay token de Hugging Face, usarlo para autenticar las descargas
model_kwargs = {
    "device_map": "auto",
    "torch_dtype": torch.bfloat16,
}
if HF_TOKEN:
    model_kwargs["token"] = HF_TOKEN

model = outlines.models.transformers_vision(
    MODEL_PATH,
    model_class=Qwen2_5_VLForConditionalGeneration,
    model_kwargs=model_kwargs,
)

# Forzar el uso del token en el procesador también si es necesario
if HF_TOKEN:
    print("Using Hugging Face token for processor...")
    print(HF_TOKEN)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, token=HF_TOKEN)
else:
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Modelo y procesador cargados.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import datetime

def ts():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f]")

@app.post("/generate")
async def generate(request: Request):
    print(ts(), "[DEBUG] Recibida petición POST /generate")
    t0 = time.monotonic()
    data = await request.json()
    print(ts(), "[DEBUG] JSON recibido:", data, f"(+{time.monotonic()-t0:.2f}s)")
    image_url = data.get("image_url")
    if image_url is None:
        image_url = "https://alonsoastroza.com/projects/ft-hackathon/avello.jpg"
    print(ts(), f"[DEBUG] Usando image_url: {image_url}")
    request_id = uuid4()
    print(ts(), f"[DEBUG] Generating response to request {request_id}")

    print(ts(), "[DEBUG] Descargando imagen y convirtiendo a PIL.Image...")
    t_img = time.monotonic()
    image = get_image_from_url(image_url)
    print(ts(), f"[DEBUG] Imagen descargada y convertida. (+{time.monotonic()-t_img:.2f}s)")

    # Set up the content you want to send to the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": f"""You are an expert at writing movie scripts based on images.\nPlease describe the scene in a movie script format. Be concise and clear.\n\nReturn the information in the following JSON schema:\n{SceneMovieScript.model_json_schema()}\n""",
                },
            ],
        }
    ]

    print(ts(), "[DEBUG] Aplicando plantilla de chat con el processor...")
    t_prompt = time.monotonic()
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(ts(), f"[DEBUG] Prompt generado. (+{time.monotonic()-t_prompt:.2f}s)")

    print(ts(), "[DEBUG] Inicializando script_generator de outlines...")
    t_gen = time.monotonic()
    script_generator = outlines.generate.json(
        model,
        SceneMovieScript,
    )
    print(ts(), f"[DEBUG] script_generator inicializado. (+{time.monotonic()-t_gen:.2f}s)")

    print(ts(), "[DEBUG] Ejecutando script_generator (esto puede demorar mucho si el modelo es grande)...")
    t_run = time.monotonic()
    result = script_generator(prompt, [image])
    print(ts(), f"[DEBUG] script_generator finalizado. (+{time.monotonic()-t_run:.2f}s)")

    total = time.monotonic() - t0
    print(ts(), f"request {request_id} completed in {total:.2f} seconds")
    print(ts(), f"Response: {result}")

    return JSONResponse({"heading": result.heading, "action": result.action, "character": result.character})

@app.post("/generate_base64")
async def generate_base64(request: Request):
    print(ts(), "[DEBUG] Recibida petición POST /generate_base64")
    t0 = time.monotonic()
    data = await request.json()
    print(ts(), "[DEBUG] JSON recibido:", data, f"(+{time.monotonic()-t0:.2f}s)")
    image_b64 = data.get("image_base64")
    if not image_b64:
        return JSONResponse({"error": "Missing image_base64 field"}, status_code=400)
    print(ts(), f"[DEBUG] Recibido base64, decodificando...")
    t_img = time.monotonic()
    try:
        img_bytes = b64decode(image_b64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(ts(), f"[ERROR] No se pudo decodificar la imagen: {e}")
        return JSONResponse({"error": "Invalid base64 image"}, status_code=400)
    print(ts(), f"[DEBUG] Imagen decodificada. (+{time.monotonic()-t_img:.2f}s)")

    # Set up the content you want to send to the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": f"""You are an expert at writing movie scripts based on images.\nPlease describe the scene in a movie script format. Be concise and clear.\n\nReturn the information in the following JSON schema:\n{SceneMovieScript.model_json_schema()}\n""",
                },
            ],
        }
    ]

    print(ts(), "[DEBUG] Aplicando plantilla de chat con el processor...")
    t_prompt = time.monotonic()
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(ts(), f"[DEBUG] Prompt generado. (+{time.monotonic()-t_prompt:.2f}s)")

    print(ts(), "[DEBUG] Inicializando script_generator de outlines...")
    t_gen = time.monotonic()
    script_generator = outlines.generate.json(
        model,
        SceneMovieScript,
    )
    print(ts(), f"[DEBUG] script_generator inicializado. (+{time.monotonic()-t_gen:.2f}s)")

    print(ts(), "[DEBUG] Ejecutando script_generator (esto puede demorar mucho si el modelo es grande)...")
    t_run = time.monotonic()
    result = script_generator(prompt, [image])
    print(ts(), f"[DEBUG] script_generator finalizado. (+{time.monotonic()-t_run:.2f}s)")

    total = time.monotonic() - t0
    print(ts(), f"request (base64) completed in {total:.2f} seconds")
    print(ts(), f"Response: {result}")

    return JSONResponse({"heading": result.heading, "action": result.action, "character": result.character})

# Para correr localmente: uvicorn scene_describer:app --reload