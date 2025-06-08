import time
from uuid import uuid4
from fastapi import Request
from fastapi.responses import JSONResponse
from base64 import b64decode
import datetime
from scene_common import get_image_from_url, FightingGameHUD

def ts():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f]")

def build_fighting_game_messages(image, model_schema):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"""
You are an expert at analyzing fighting game HUDs (Heads-Up Displays) from images.
Please extract and describe the following HUD elements for each player, being as specific as possible:

For **Player 1** (left) and **Player 2** (right), provide:
- character: Name of the character for this player.
- health: Current health/stamina of the player as a percentage or value.
- ratio: Ratio number for the current character (1-4).
- guard_gauge: Guard gauge status for the player.
- position: Bounding box of the player in the format [x_min, y_min, x_max, y_max] in image coordinates.

Also provide:
- round_timer: The round timer is a number (integer) located in the upper middle of the screen, and shows how much time is left in the round. The timer starts from 999 and counts down to 0. When the timer hits 0, whichever character has more health remaining wins the round and keeps all of that health in the next round. Only return the number, not a string.
- combo_counter_messages: The numbers show how many hits you have done in a combo. Below it, there can also appear messages like 'Guard Crush', 'Reversal', 'Counter', and the type of K.O. in the end of a round.

Return the information in the following JSON schema:
{model_schema}
"""},
            ],
        }
    ]


def api_generate(processor, model, model_schema):
    async def endpoint(request: Request):
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

        messages = build_fighting_game_messages(image, model_schema.model_json_schema())

        print(ts(), "[DEBUG] Aplicando plantilla de chat con el processor...")
        t_prompt = time.monotonic()
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(ts(), f"[DEBUG] Prompt generado. (+{time.monotonic()-t_prompt:.2f}s)")

        print(ts(), "[DEBUG] Inicializando script_generator de outlines...")
        t_gen = time.monotonic()
        import outlines
        script_generator = outlines.generate.json(
            model,
            model_schema,
        )
        print(ts(), f"[DEBUG] script_generator inicializado. (+{time.monotonic()-t_gen:.2f}s)")

        print(ts(), "[DEBUG] Ejecutando script_generator...")
        t_run = time.monotonic()
        result = script_generator(prompt, [image])
        print(ts(), f"[DEBUG] script_generator finalizado. (+{time.monotonic()-t_run:.2f}s)")

        total = time.monotonic() - t0
        print(ts(), f"request {request_id} completed in {total:.2f} seconds")
        print(ts(), f"Response: {result}")

        # Save response to a timestamped JSON file
        import json
        import os
        output_dir = "responses"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = os.path.join(output_dir, f"response_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
        print(ts(), f"Response saved to {output_path}")

        return JSONResponse(result.model_dump())
    return endpoint


def api_generate_base64(processor, model, model_schema):
    async def endpoint(request: Request):
        print(ts(), "[DEBUG] Recibida petición POST /generate_base64")
        t0 = time.monotonic()
        data = await request.json()
        print(ts(), "[DEBUG] JSON recibido:", data, f"(+{time.monotonic()-t0:.2f}s)")
        image_b64 = data.get("image_base64")
        if not image_b64:
            return JSONResponse({"error": "Missing image_base64 field"}, status_code=400)
        print(ts(), f"[DEBUG] Recibido base64, decodificando...")
        t_img = time.monotonic()
        from PIL import Image
        from io import BytesIO
        try:
            img_bytes = b64decode(image_b64)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(ts(), f"[ERROR] No se pudo decodificar la imagen: {e}")
            return JSONResponse({"error": "Invalid base64 image"}, status_code=400)
        print(ts(), f"[DEBUG] Imagen decodificada. (+{time.monotonic()-t_img:.2f}s)")

        messages = build_fighting_game_messages(image, model_schema.model_json_schema())

        print(ts(), "[DEBUG] Aplicando plantilla de chat con el processor...")
        t_prompt = time.monotonic()
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(ts(), f"[DEBUG] Prompt generado. (+{time.monotonic()-t_prompt:.2f}s)")

        print(ts(), "[DEBUG] Inicializando script_generator de outlines...")
        t_gen = time.monotonic()
        import outlines
        script_generator = outlines.generate.json(
            model,
            model_schema,
        )
        print(ts(), f"[DEBUG] script_generator inicializado. (+{time.monotonic()-t_gen:.2f}s)")

        print(ts(), "[DEBUG] Ejecutando script_generator...")
        t_run = time.monotonic()
        result = script_generator(prompt, [image])
        print(ts(), f"[DEBUG] script_generator finalizado. (+{time.monotonic()-t_run:.2f}s)")

        total = time.monotonic() - t0
        print(ts(), f"request (base64) completed in {total:.2f} seconds")
        print(ts(), f"Response: {result}")

        # Save response to a timestamped JSON file
        import json
        import os
        output_dir = "responses"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = os.path.join(output_dir, f"response_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
        print(ts(), f"Response saved to {output_path}")

        return JSONResponse(result.model_dump())
    return endpoint
