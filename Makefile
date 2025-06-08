# Makefile para lanzar el servidor FastAPI de scene_describer.py

run:
	. .venv/Scripts/activate && uvicorn scene_describer:app --reload --host 0.0.0.0 --port 8000
