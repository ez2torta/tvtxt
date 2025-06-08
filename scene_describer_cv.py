import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scene_common import analyze_fighting_game_scene_cv

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze_cv")
def analyze_cv_endpoint(image_path: str = "EO_inline1.jpg"):
    """
    Analiza una imagen de juego de pelea usando computer vision clásico (OpenCV + Tesseract).
    Por defecto usa EO_inline1.jpg en el root del proyecto.
    """
    result = analyze_fighting_game_scene_cv(image_path)
    return JSONResponse(content=result)

# Para correr localmente: uvicorn scene_describer_cv:app --reload
