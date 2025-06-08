import os
import time
from uuid import uuid4
from typing import List
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
from pydantic import BaseModel, Field

# --- Computer Vision imports ---
import cv2
import numpy as np
import pytesseract

# --- Computer Vision HUD analysis ---
def analyze_fighting_game_scene_cv(image_path):
    # Inicializar debug_dir antes de cualquier uso
    debug_dir = "debug_cv"
    os.makedirs(debug_dir, exist_ok=True)
    debug_data = {"input_image": image_path}

    # Cargar imagen
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"No se pudo cargar la imagen: {image_path}"}

    height, width, _ = img.shape

    # --- Timer (reloj de round) ---
    # Asumimos que el timer está centrado en la parte superior, ocupa ~10-15% del ancho y ~6-10% de la altura
    timer_w = int(0.13 * width)
    timer_h = int(0.09 * height)
    timer_x = (width - timer_w) // 2
    timer_y = int(0.01 * height)
    timer_region = img[timer_y:timer_y+timer_h, timer_x:timer_x+timer_w]
    cv2.imwrite(os.path.join(debug_dir, "timer_region.jpg"), timer_region)
    # Preprocesado para OCR: gris, umbralizado
    timer_gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
    _, timer_bin = cv2.threshold(timer_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(debug_dir, "timer_region_bin.jpg"), timer_bin)
    # OCR restrictivo: solo dígitos, psm 7 (una línea)
    timer_ocr = pytesseract.image_to_string(timer_bin, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    import re
    timer_digits = re.findall(r'\d{2,3}', timer_ocr)
    if timer_digits:
        round_timer = int(timer_digits[0])
    else:
        round_timer = None
    """
    Analiza una imagen de un juego de pelea para detectar una barra de vida (por color)
    y extraer texto usando OCR (Tesseract).
    """

    import json
    import datetime
    debug_dir = "debug_cv"
    os.makedirs(debug_dir, exist_ok=True)
    debug_data = {"input_image": image_path}

    # Cargar imagen
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"No se pudo cargar la imagen: {image_path}"}

    height, width, _ = img.shape
    # Ajustar: que la barra empiece un poco más abajo y sea un poco más baja
    bar_y_start = int(0.05 * height)  # Empieza al 5% de la altura
    bar_y_end = int(0.17 * height)    # Termina al 17% de la altura (barra más delgada y desplazada)
    bar_height = bar_y_end - bar_y_start
    bar_region = img[bar_y_start:bar_y_end, :]
    player1_region = bar_region[:, 0:int(0.45*width)]
    player2_region = bar_region[:, int(0.55*width):width]

    # Guardar imágenes intermedias para debugging
    cv2.imwrite(os.path.join(debug_dir, "bar_region.jpg"), bar_region)
    cv2.imwrite(os.path.join(debug_dir, "player1_region.jpg"), player1_region)
    cv2.imwrite(os.path.join(debug_dir, "player2_region.jpg"), player2_region)

    def get_bar_info(region, full_img, x_offset, y_offset, player_label):
        # --- Barra de vida discontinua ---
        region_proc = region.copy()
        hsv = cv2.cvtColor(region_proc, cv2.COLOR_BGR2HSV)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_region_hsv.jpg"), hsv)
        # Amarillo: vida alta
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Naranja: vida media
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([20, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        # Rojo: vida baja
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        # Sumar todos los píxeles de cada color
        total_bar_pixels = region.shape[0] * region.shape[1]
        yellow_pixels = cv2.countNonZero(mask_yellow)
        orange_pixels = cv2.countNonZero(mask_orange)
        red_pixels = cv2.countNonZero(mask_red)
        bar_pixels = yellow_pixels + orange_pixels + red_pixels
        # Paso 1: máscaras de color
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_yellow.jpg"), mask_yellow)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([20, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_orange.jpg"), mask_orange)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_red.jpg"), mask_red)

        # Paso 2: máscara combinada
        mask_bar = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, mask_red))
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_bar.jpg"), mask_bar)
        total_bar_pixels = region.shape[0] * region.shape[1]

        # --- MÉTODO ALTERNATIVO: segmento horizontal más largo por fila (robusto a manchas verticales, SOLO MITAD INFERIOR) ---
        mask_bar_bin = (mask_bar > 0).astype(np.uint8)
        bar_cols = mask_bar.shape[1]
        bar_rows = mask_bar.shape[0]
        start_row = bar_rows // 2
        max_segments = []
        for row in range(start_row, bar_rows):
            row_data = mask_bar_bin[row, :]
            # Buscar el segmento más largo de 1s en la fila
            max_len = 0
            curr_len = 0
            for val in row_data:
                if val:
                    curr_len += 1
                    if curr_len > max_len:
                        max_len = curr_len
                else:
                    curr_len = 0
            max_segments.append(max_len)
        # Tomar el máximo segmento horizontal (o promedio de los N mayores para robustez)
        if len(max_segments) > 0:
            max_segment = max(max_segments)
            avg_top5 = np.mean(sorted(max_segments, reverse=True)[:5])
            health_bar_ratio_segment = max_segment / bar_cols if bar_cols > 0 else 0
            health_bar_ratio_segment_avg5 = avg_top5 / bar_cols if bar_cols > 0 else 0
        else:
            health_bar_ratio_segment = 0
            health_bar_ratio_segment_avg5 = 0
        # Guardar debug visual del segmento más largo (solo mitad inferior)
        debug_segment_img = np.stack([mask_bar_bin*255]*3, axis=-1)
        for idx, row in enumerate(range(start_row, bar_rows)):
            seglen = max_segments[idx]
            if seglen > 0:
                # Buscar inicio y fin del segmento más largo en la fila
                row_data = mask_bar_bin[row, :]
                curr_len = 0
                max_len = 0
                max_start = 0
                curr_start = 0
                for i, val in enumerate(row_data):
                    if val:
                        if curr_len == 0:
                            curr_start = i
                        curr_len += 1
                        if curr_len > max_len:
                            max_len = curr_len
                            max_start = curr_start
                    else:
                        curr_len = 0
                # Pintar el segmento más largo en azul SOLO en la mitad inferior
                debug_segment_img[row, max_start:max_start+max_len] = (255, 0, 0)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_bar_longest_segment.jpg"), debug_segment_img)
        # Guardar debug numérico del método alternativo
        with open(os.path.join(debug_dir, f"{player_label}_bar_segment_debug.txt"), "w") as f:
            f.write(f"max_segments: {max_segments}\n")
            f.write(f"max_segment: {max_segment}\n")
            f.write(f"avg_top5: {avg_top5}\n")
            f.write(f"health_bar_ratio_segment: {health_bar_ratio_segment}\n")
            f.write(f"health_bar_ratio_segment_avg5: {health_bar_ratio_segment_avg5}\n")
        # --- FIN MÉTODO ALTERNATIVO ---
        region_proc = region.copy()
        hsv = cv2.cvtColor(region_proc, cv2.COLOR_BGR2HSV)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_region_hsv.jpg"), hsv)
        # Amarillo: vida alta
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Naranja: vida media
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([20, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        # Rojo: vida baja
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        # Sumar todos los píxeles de cada color
        total_bar_pixels = region.shape[0] * region.shape[1]
        yellow_pixels = cv2.countNonZero(mask_yellow)
        orange_pixels = cv2.countNonZero(mask_orange)
        red_pixels = cv2.countNonZero(mask_red)
        bar_pixels = yellow_pixels + orange_pixels + red_pixels
        # Paso 1: máscaras de color
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_yellow.jpg"), mask_yellow)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([20, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_orange.jpg"), mask_orange)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_red.jpg"), mask_red)

        # Paso 2: máscara combinada
        mask_bar = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, mask_red))
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_bar.jpg"), mask_bar)
        total_bar_pixels = region.shape[0] * region.shape[1]

        # Paso 3: análisis columna por columna (debug visual y numérico)
        bar_cols = mask_bar.shape[1]
        bar_rows = mask_bar.shape[0]
        threshold = int(0.5 * bar_rows)  # Al menos 50% de la altura debe estar "llena" en la columna
        debug_col_array = np.zeros((bar_rows, bar_cols, 3), dtype=np.uint8)  # Para visualización en color
        col_filled = np.zeros(bar_cols, dtype=bool)
        if player_label == "player1":
            last_filled = -1
            for col in range(bar_cols):
                if np.count_nonzero(mask_bar[:, col]) >= threshold:
                    last_filled = col
                    col_filled[col] = True
                    debug_col_array[:, col] = (0, 255, 0)  # Verde para columnas llenas
                else:
                    debug_col_array[:, col] = (0, 0, 255)  # Rojo para vacías
            health_bar_ratio = (last_filled + 1) / bar_cols if bar_cols > 0 and last_filled >= 0 else 0
            # Línea azul en el límite detectado
            if last_filled >= 0:
                cv2.line(debug_col_array, (last_filled, 0), (last_filled, bar_rows-1), (255, 0, 0), 2)
        else:
            first_filled = bar_cols
            for col in range(bar_cols-1, -1, -1):
                if np.count_nonzero(mask_bar[:, col]) >= threshold:
                    first_filled = col
                    col_filled[col] = True
                    debug_col_array[:, col] = (0, 255, 0)
                else:
                    debug_col_array[:, col] = (0, 0, 255)
            health_bar_ratio = (bar_cols - first_filled) / bar_cols if bar_cols > 0 and first_filled < bar_cols else 0
            # Línea azul en el límite detectado
            if first_filled < bar_cols:
                cv2.line(debug_col_array, (first_filled, 0), (first_filled, bar_rows-1), (255, 0, 0), 2)
        # Paso 4: guardar debug de columnas llenas (color)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_bar_cols_filled_color.jpg"), debug_col_array)
        # También guardar la versión binaria (como antes)
        debug_col_array_bin = np.where(col_filled, 255, 0).astype(np.uint8)
        debug_col_array_bin = np.tile(debug_col_array_bin, (bar_rows, 1))
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_mask_bar_cols_filled.jpg"), debug_col_array_bin)
        # Guardar valores numéricos de debug
        with open(os.path.join(debug_dir, f"{player_label}_bar_debug.txt"), "w") as f:
            f.write(f"bar_cols: {bar_cols}\n")
            f.write(f"bar_rows: {bar_rows}\n")
            f.write(f"threshold: {threshold}\n")
            f.write(f"col_filled: {col_filled.tolist()}\n")
            if player_label == "player1":
                f.write(f"last_filled: {last_filled}\n")
            else:
                f.write(f"first_filled: {first_filled}\n")
            f.write(f"health_bar_ratio: {health_bar_ratio}\n")

        # Porcentajes de cada color (suma total de píxeles)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        orange_pixels = cv2.countNonZero(mask_orange)
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pct = yellow_pixels / total_bar_pixels if total_bar_pixels > 0 else 0
        orange_pct = orange_pixels / total_bar_pixels if total_bar_pixels > 0 else 0
        red_pct = red_pixels / total_bar_pixels if total_bar_pixels > 0 else 0
        # Estado de vida
        if yellow_pct > orange_pct and yellow_pct > red_pct:
            health_state = "high"
        elif orange_pct > red_pct:
            health_state = "medium"
        else:
            health_state = "low"
        # Guardar debug numérico de colores
        with open(os.path.join(debug_dir, f"{player_label}_bar_colors_debug.txt"), "w") as f:
            f.write(f"yellow_pixels: {yellow_pixels}\n")
            f.write(f"orange_pixels: {orange_pixels}\n")
            f.write(f"red_pixels: {red_pixels}\n")
            f.write(f"yellow_pct: {yellow_pct}\n")
            f.write(f"orange_pct: {orange_pct}\n")
            f.write(f"red_pct: {red_pct}\n")
            f.write(f"health_state: {health_state}\n")

        # OCR: nombre del personaje (desplazamos la franja un poco más abajo)
        name_offset = int(0.03 * full_img.shape[0])  # Desplazamiento extra hacia abajo
        name_y_start = min(y_offset + region.shape[0] + name_offset, full_img.shape[0]-1)
        name_y_end = min(name_y_start + int(0.07*full_img.shape[0]), full_img.shape[0])
        name_region = full_img[name_y_start:name_y_end, x_offset:x_offset+region.shape[1]]
        # Dividir la región en dos mitades: una para el nombre, otra para la guardia
        mid_x = region.shape[1] // 2
        if player_label == "player1":
            name_half = name_region[:, :mid_x]
            guard_half = name_region[:, mid_x:]
        else:  # player2
            guard_half = name_region[:, :mid_x]
            name_half = name_region[:, mid_x:]
        # Guardar imágenes para debugging
        if name_region.size > 0:
            cv2.imwrite(os.path.join(debug_dir, f"{player_label}_name_region.jpg"), name_region)
        if name_half.size > 0:
            cv2.imwrite(os.path.join(debug_dir, f"{player_label}_name_half.jpg"), name_half)
        if guard_half.size > 0:
            cv2.imwrite(os.path.join(debug_dir, f"{player_label}_guard_region.jpg"), guard_half)
        # OCR nombre
        if name_half.size > 0:
            name_half_rgb = cv2.cvtColor(name_half, cv2.COLOR_BGR2RGB)
            name_ocr = pytesseract.image_to_string(name_half_rgb)
        else:
            name_ocr = ""
        # Guard gauge: análisis robusto de barra de guardia (verde)
        if guard_half.size > 0:
            guard_half_hsv = cv2.cvtColor(guard_half, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 80, 80])
            upper_green = np.array([80, 255, 255])
            mask_guard = cv2.inRange(guard_half_hsv, lower_green, upper_green)
            # Análisis columna por columna
            guard_cols = mask_guard.shape[1]
            guard_rows = mask_guard.shape[0]
            threshold = int(0.5 * guard_rows)  # Al menos 50% de la altura debe estar "llena" en la columna
            if player_label == "player1":
                last_filled = -1
                for col in range(guard_cols):
                    if np.count_nonzero(mask_guard[:, col]) >= threshold:
                        last_filled = col
                guard_percent = int(100 * (last_filled + 1) / guard_cols) if guard_cols > 0 and last_filled >= 0 else 0
            else:
                first_filled = guard_cols
                for col in range(guard_cols-1, -1, -1):
                    if np.count_nonzero(mask_guard[:, col]) >= threshold:
                        first_filled = col
                guard_percent = int(100 * (guard_cols - first_filled) / guard_cols) if guard_cols > 0 and first_filled < guard_cols else 0
            guard_half_rgb = cv2.cvtColor(guard_half, cv2.COLOR_BGR2RGB)
            guard_ocr = pytesseract.image_to_string(guard_half_rgb)
        else:
            guard_ocr = ""
            guard_percent = None

        # Ratio: identificar por color dominante en la región
        # Ratio: identificar por color dominante en la región
        ratio_width = int(0.12 * full_img.shape[1])
        ratio_height = region.shape[0]
        ratio_y_start = y_offset
        ratio_y_end = min(y_offset + ratio_height, full_img.shape[0])
        if player_label == "player1":
            ratio_x_start = max(x_offset, 0)
            ratio_x_end = min(x_offset + ratio_width, full_img.shape[1])
        else:  # player2
            ratio_x_end = full_img.shape[1]
            ratio_x_start = max(ratio_x_end - ratio_width, 0)
        ratio_region = full_img[ratio_y_start:ratio_y_end, ratio_x_start:ratio_x_end]
        ratio_detected = ""
        if ratio_region.size > 0:
            cv2.imwrite(os.path.join(debug_dir, f"{player_label}_ratio_region.jpg"), ratio_region)
            ratio_hsv = cv2.cvtColor(ratio_region, cv2.COLOR_BGR2HSV)
            # Definir rangos HSV para cada ratio
            # Ratio 4: púrpura
            lower_purple = np.array([130, 80, 80])
            upper_purple = np.array([155, 255, 255])
            # Ratio 3: celeste
            lower_cyan = np.array([85, 80, 80])
            upper_cyan = np.array([105, 255, 255])
            # Ratio 2: verde
            lower_green = np.array([40, 80, 80])
            upper_green = np.array([80, 255, 255])
            # Ratio 1: amarillo
            lower_yellow_r = np.array([20, 100, 100])
            upper_yellow_r = np.array([35, 255, 255])
            mask_purple = cv2.inRange(ratio_hsv, lower_purple, upper_purple)
            mask_cyan = cv2.inRange(ratio_hsv, lower_cyan, upper_cyan)
            mask_green = cv2.inRange(ratio_hsv, lower_green, upper_green)
            mask_yellow = cv2.inRange(ratio_hsv, lower_yellow_r, upper_yellow_r)
            counts = {
                "4": cv2.countNonZero(mask_purple),
                "3": cv2.countNonZero(mask_cyan),
                "2": cv2.countNonZero(mask_green),
                "1": cv2.countNonZero(mask_yellow),
            }
            # Elegir el ratio con más píxeles
            max_ratio = max(counts, key=counts.get)
            if counts[max_ratio] > 0:
                ratio_detected = max_ratio
            else:
                # fallback a OCR si no hay color dominante
                ratio_region_rgb = cv2.cvtColor(ratio_region, cv2.COLOR_BGR2RGB)
                ratio_ocr_raw = pytesseract.image_to_string(ratio_region_rgb, config='--psm 8 -c tessedit_char_whitelist=1234')
                import re
                match = re.search(r'[1-4]', ratio_ocr_raw)
                ratio_detected = match.group(0) if match else ""
        else:
            ratio_detected = ""

        # OCR de la barra (por si hay texto relevante)
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        ocr_text = pytesseract.image_to_string(region_rgb)
        cv2.imwrite(os.path.join(debug_dir, f"{player_label}_region_rgb.jpg"), region)

        return {
            # Usar el método alternativo (mitad inferior) como valor principal
            "health_bar_ratio": health_bar_ratio_segment,
            "health_bar_ratio_segment": health_bar_ratio_segment,
            "health_bar_ratio_segment_avg5": health_bar_ratio_segment_avg5,
            "health_bar_ratio_colwise": health_bar_ratio,
            "health_bar_yellow_pct": yellow_pct,
            "health_bar_orange_pct": orange_pct,
            "health_bar_red_pct": red_pct,
            "health_state": health_state,
            "ocr_text": ocr_text.strip(),
            "character_name": name_ocr.strip(),
            "guard_gauge": guard_percent,
            "guard_gauge_ocr": guard_ocr.strip(),
            "ratio": ratio_detected.strip(),
        }

    player1_info = get_bar_info(player1_region, img, 0, 0, "player1")
    player2_info = get_bar_info(player2_region, img, int(0.55*width), 0, "player2")

    output = {
        "player1": {
            **player1_info,
            "region_analyzed": {
                "x_min": 0,
                "y_min": 0,
                "x_max": int(0.45*width),
                "y_max": bar_height
            }
        },
        "player2": {
            **player2_info,
            "region_analyzed": {
                "x_min": int(0.55*width),
                "y_min": 0,
                "x_max": width,
                "y_max": bar_height
            }
        },
        "round_timer": round_timer
    }

    # Guardar el JSON de salida para debugging
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = os.path.join(debug_dir, f"output_{now}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[DEBUG] analyze_fighting_game_scene_cv: output saved to {json_path}")
    return output

# Shared utility to get image from URL

def get_image_from_url(image_url):
    img_byte_stream = BytesIO(urlopen(image_url).read())
    return Image.open(img_byte_stream).convert("RGB")

# Shared Pydantic models

class PlayerHUD(BaseModel):
    character: str = Field(..., description="Name of the character for this player.")
    health: str = Field(..., description="Health bar for the player, usually in a solid color which can be yellow (full or almost full) to red (dangerously low).")
    ratio: int = Field(..., ge=1, le=4, description="Ratio number for the current character (1-4).")
    guard_gauge: str = Field(..., description="Small bar below the health bar that shows the guard gauge status for the player.")
    position: List[int] = Field(..., description="Bounding box of the player in the format [x_min, y_min, x_max, y_max] in image coordinates.")


class FightingGameHUD(BaseModel):
    player1: PlayerHUD = Field(..., description="HUD information for the player on the left (Player 1).")
    player2: PlayerHUD = Field(..., description="HUD information for the player on the right (Player 2).")
    round_timer: int = Field(..., ge=0, le=999, description="This timer is in the upper middle of the screen and shows how much time is left in the round.")
    combo_counter_messages: str = Field(..., description="The numbers show how many hits you have done in a combo. Below it, there can also appear messages like 'Guard Crush', 'Reversal', 'Counter', and the type of K.O. in the end of a round.")
