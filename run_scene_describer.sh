#!/bin/sh
# Script para lanzar el servidor FastAPI de scene_describer.py con recarga automática

# Activa el entorno virtual si existe
if [ -d ".venv" ]; then
    . .venv/bin/activate
fi

# Ejecuta uvicorn
uvicorn scene_describer:app --reload --host 0.0.0.0 --port 8000
