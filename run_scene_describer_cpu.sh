#!/bin/bash
# Script para lanzar el servidor FastAPI en modo CPU (scene_describer_cpu.py)

# Activa el entorno virtual si existe
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Lanza el servidor en modo CPU
uvicorn scene_describer_cpu:app --host 0.0.0.0 --port 8000 --reload
