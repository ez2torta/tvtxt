
import os
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import outlines
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv

from scene_common import get_image_from_url, FightingGameHUD
from scene_api import api_generate, api_generate_base64

# Cargar variables de entorno desde .env si existe
load_dotenv()

# MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2-VL-7B-Instruct")
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" # probando ahora que tan veloz puede ser
# MODEL_PATH = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
HF_TOKEN = os.environ.get("HF_TOKEN")



from scene_common import FightingGameHUD

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

# Delegate endpoints to shared API logic
app.post("/generate")(api_generate(processor, model, FightingGameHUD))
app.post("/generate_base64")(api_generate_base64(processor, model, FightingGameHUD))

# Para correr localmente: uvicorn scene_describer:app --reload