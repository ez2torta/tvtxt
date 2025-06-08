
import os
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import outlines
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv
import datetime

from scene_common import get_image_from_url, FightingGameHUD
from scene_api import api_generate, api_generate_base64


def ts():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f]")

# Cargar variables de entorno desde .env si existe
load_dotenv()

MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")



print("Cargando modelo y procesador SOLO CPU...")
model_kwargs = {
    "device_map": "cpu",
    "torch_dtype": "float32",
}
if HF_TOKEN:
    model_kwargs["token"] = HF_TOKEN

model = outlines.models.transformers_vision(
    MODEL_PATH,
    model_class=Qwen2_5_VLForConditionalGeneration,
    model_kwargs=model_kwargs,
)

if HF_TOKEN:
    print("Using Hugging Face token for processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, token=HF_TOKEN)
else:
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Modelo y procesador cargados (CPU).")


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


# Para correr localmente: uvicorn scene_describer_cpu:app --reload
