#!/bin/bash
# Script para levantar la API de Computer Vision clásica (OpenCV + Tesseract)

uvicorn scene_describer_cv:app --reload
