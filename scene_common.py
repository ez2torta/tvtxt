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

import json
import datetime


def save_debug_image(path, img):
    """Guarda una imagen de debug si la ruta y la imagen son válidas."""
    if img is not None and path:
        cv2.imwrite(path, img)


def ocr_region(region, config="", debug_path=None):
    """Realiza OCR sobre una región (con preprocesado básico) y guarda debug si se indica."""
    if region is None or region.size == 0:
        return ""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if debug_path:
        save_debug_image(debug_path, bin_img)
    return pytesseract.image_to_string(bin_img, config=config)


def segment_health_colors(hsv):
    """Devuelve las máscaras de amarillo, naranja y rojo para la barra de vida."""
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([20, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    return mask_yellow, mask_orange, mask_red


def analyze_health_bar(mask_bar, debug_dir, player_label):
    """Calcula los ratios de vida usando ambos métodos y guarda debug visual."""
    mask_bar_bin = (mask_bar > 0).astype(np.uint8)
    bar_cols = mask_bar.shape[1]
    bar_rows = mask_bar.shape[0]
    start_row = bar_rows // 2
    max_segments = []
    for row in range(start_row, bar_rows):
        row_data = mask_bar_bin[row, :]
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
    if len(max_segments) > 0:
        max_segment = max(max_segments)
        avg_top5 = np.mean(sorted(max_segments, reverse=True)[:5])
        health_bar_ratio_segment = max_segment / bar_cols if bar_cols > 0 else 0
        health_bar_ratio_segment_avg5 = avg_top5 / bar_cols if bar_cols > 0 else 0
    else:
        health_bar_ratio_segment = 0
        health_bar_ratio_segment_avg5 = 0
    debug_segment_img = np.stack([mask_bar_bin * 255] * 3, axis=-1)
    for idx, row in enumerate(range(start_row, bar_rows)):
        seglen = max_segments[idx]
        if seglen > 0:
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
            debug_segment_img[row, max_start : max_start + max_len] = (255, 0, 0)
    save_debug_image(
        os.path.join(debug_dir, f"{player_label}_mask_bar_longest_segment.jpg"),
        debug_segment_img,
    )
    return health_bar_ratio_segment, health_bar_ratio_segment_avg5


# --- Computer Vision HUD analysis ---
def extract_timer(img, debug_dir):
    """Extrae el timer (reloj de round) de la parte superior central de la imagen."""
    height, width = img.shape[:2]
    timer_w = int(0.13 * width) + int(0.05 * width)  # Crece 5% a la derecha
    timer_h = int(0.09 * height) + int(0.05 * height)  # Crece 5% hacia abajo
    timer_x = (width - timer_w) // 2
    timer_y = int(0.01 * height) + int(
        0.10 * height
    )  # Desplaza hacia abajo un 10% de la altura de la imagen
    # Encoge 4 píxeles por cada lado (arriba, abajo, izquierda, derecha)
    shrink = 4
    x1 = max(timer_x + shrink, 0)
    y1 = max(timer_y + shrink, 0)
    x2 = min(timer_x + timer_w - shrink, width)
    y2 = min(timer_y + timer_h - shrink, height)
    timer_region = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(debug_dir, "timer_region.jpg"), timer_region)
    timer_gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
    _, timer_bin = cv2.threshold(
        timer_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Invertir si la mayoría de píxeles son negros (números negros con borde blanco)
    if np.mean(timer_bin) < 127:
        timer_bin_inv = cv2.bitwise_not(timer_bin)
        cv2.imwrite(os.path.join(debug_dir, "timer_region_bin_inv.jpg"), timer_bin_inv)
        timer_ocr = pytesseract.image_to_string(
            timer_bin_inv, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
    else:
        cv2.imwrite(os.path.join(debug_dir, "timer_region_bin.jpg"), timer_bin)
        timer_ocr = pytesseract.image_to_string(
            timer_bin, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
    import re

    timer_digits = re.findall(r"\d{2,3}", timer_ocr)
    if timer_digits:
        return int(timer_digits[0])
    else:
        return None


def analyze_fighting_game_scene_cv(image_path):

    debug_dir = "debug_cv"
    os.makedirs(debug_dir, exist_ok=True)
    debug_data = {"input_image": image_path}

    # Cargar imagen
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"No se pudo cargar la imagen: {image_path}"}

    height, width, _ = img.shape

    # --- Barras de super (zona inferior) ---
    super_y_start = int(0.89 * height)
    super_y_end = int(0.98 * height)
    super_height = super_y_end - super_y_start
    super_region = img[super_y_start:super_y_end, :]
    # Player 1: izquierda, Player 2: derecha
    super1_region = super_region[:, 0 : int(0.45 * width)]
    super2_region = super_region[:, int(0.55 * width) : width]
    save_debug_image(os.path.join(debug_dir, "super1_region.jpg"), super1_region)
    save_debug_image(os.path.join(debug_dir, "super2_region.jpg"), super2_region)

    def analyze_super_bar(region, player_label):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Verde (C-GROOVE)
        lower_green = np.array([45, 120, 60])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # Celeste (P-GROOVE)
        lower_cyan = np.array([85, 80, 80])
        upper_cyan = np.array([105, 255, 255])
        mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
        total_pixels = region.shape[0] * region.shape[1]
        green_pct = (
            cv2.countNonZero(mask_green) / total_pixels if total_pixels > 0 else 0
        )
        cyan_pct = cv2.countNonZero(mask_cyan) / total_pixels if total_pixels > 0 else 0
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_super_mask_green.jpg"), mask_green
        )
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_super_mask_cyan.jpg"), mask_cyan
        )
        return green_pct, cyan_pct

    # --- Timer (reloj de round) ---
    round_timer = extract_timer(img, debug_dir)

    # --- Barra de vida y HUD ---
    bar_y_start = int(0.05 * height)
    bar_y_end = int(0.17 * height)
    bar_height = bar_y_end - bar_y_start
    bar_region = img[bar_y_start:bar_y_end, :]
    player1_region = bar_region[:, 0 : int(0.45 * width)]
    player2_region = bar_region[:, int(0.55 * width) : width]

    save_debug_image(os.path.join(debug_dir, "bar_region.jpg"), bar_region)
    save_debug_image(os.path.join(debug_dir, "player1_region.jpg"), player1_region)
    save_debug_image(os.path.join(debug_dir, "player2_region.jpg"), player2_region)

    def extract_name_and_guard_regions(
        full_img, region, x_offset, y_offset, player_label
    ):
        name_offset = int(0.03 * full_img.shape[0])
        name_y_start = min(
            y_offset + region.shape[0] + name_offset, full_img.shape[0] - 1
        )
        name_y_end = min(
            name_y_start + int(0.07 * full_img.shape[0]), full_img.shape[0]
        )
        name_region = full_img[
            name_y_start:name_y_end, x_offset : x_offset + region.shape[1]
        ]
        mid_x = region.shape[1] // 2
        if player_label == "player1":
            name_half = name_region[:, :mid_x]
            guard_half = name_region[:, mid_x:]
        else:
            guard_half = name_region[:, :mid_x]
            name_half = name_region[:, mid_x:]
        return name_region, name_half, guard_half

    def extract_guard_gauge(guard_half, player_label, debug_dir):
        if guard_half.size == 0:
            return None, ""
        guard_half_hsv = cv2.cvtColor(guard_half, cv2.COLOR_BGR2HSV)
        # Ajuste automático de rango verde: probar varios rangos y elegir el que más píxeles verdes detecte
        # Filtro de saturación mínima y exclusión de rojo
        green_ranges = [
            (np.array([45, 120, 60]), np.array([85, 255, 255])),  # verde puro, saturado
            (np.array([40, 80, 80]), np.array([80, 255, 255])),  # original
            (np.array([50, 100, 100]), np.array([85, 255, 255])),  # más saturado
            (np.array([60, 50, 50]), np.array([90, 255, 255])),  # más claro
        ]
        best_mask = None
        best_count = 0
        for idx, (low, up) in enumerate(green_ranges):
            mask = cv2.inRange(guard_half_hsv, low, up)
            # Filtro de saturación mínima
            s_channel = guard_half_hsv[:, :, 1]
            mask = cv2.bitwise_and(mask, cv2.inRange(s_channel, 80, 255))
            # Excluir píxeles con alto valor en canal rojo (en BGR)
            guard_half_bgr = cv2.cvtColor(guard_half_hsv, cv2.COLOR_HSV2BGR)
            red_channel = guard_half_bgr[:, :, 2]
            mask = cv2.bitwise_and(mask, cv2.inRange(red_channel, 0, 180))
            count = cv2.countNonZero(mask)
            save_debug_image(
                os.path.join(debug_dir, f"{player_label}_mask_guard_try{idx}.jpg"), mask
            )
            if count > best_count:
                best_count = count
                best_mask = mask
        mask_guard = best_mask
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_mask_guard.jpg"), mask_guard
        )
        guard_cols = mask_guard.shape[1]
        guard_rows = mask_guard.shape[0]
        threshold = int(0.2 * guard_rows)  # Más sensible para barras delgadas
        # Método colwise (legacy)
        if player_label == "player1":
            last_filled = -1
            for col in range(guard_cols):
                if np.count_nonzero(mask_guard[:, col]) >= threshold:
                    last_filled = col
            guard_percent_colwise = (
                int(100 * (last_filled + 1) / guard_cols)
                if guard_cols > 0 and last_filled >= 0
                else 0
            )
        else:
            first_filled = guard_cols
            for col in range(guard_cols - 1, -1, -1):
                if np.count_nonzero(mask_guard[:, col]) >= threshold:
                    first_filled = col
            guard_percent_colwise = (
                int(100 * (guard_cols - first_filled) / guard_cols)
                if guard_cols > 0 and first_filled < guard_cols
                else 0
            )

        # Método robusto: segmento horizontal más largo en la mitad inferior
        mask_guard_bin = (mask_guard > 0).astype(np.uint8)
        start_row = guard_rows // 2
        max_segments = []
        for row in range(start_row, guard_rows):
            row_data = mask_guard_bin[row, :]
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
        if len(max_segments) > 0:
            max_segment = max(max_segments)
            avg_top5 = np.mean(sorted(max_segments, reverse=True)[:5])
            guard_gauge_ratio_segment = (
                max_segment / guard_cols if guard_cols > 0 else 0
            )
            guard_gauge_ratio_segment_avg5 = (
                avg_top5 / guard_cols if guard_cols > 0 else 0
            )
        else:
            guard_gauge_ratio_segment = 0
            guard_gauge_ratio_segment_avg5 = 0
        # Debug visual del segmento más largo
        debug_segment_img = np.stack([mask_guard_bin * 255] * 3, axis=-1)
        for idx, row in enumerate(range(start_row, guard_rows)):
            seglen = max_segments[idx]
            if seglen > 0:
                row_data = mask_guard_bin[row, :]
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
                debug_segment_img[row, max_start : max_start + max_len] = (255, 0, 0)
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_mask_guard_longest_segment.jpg"),
            debug_segment_img,
        )
        # Debug numérico
        with open(
            os.path.join(debug_dir, f"{player_label}_guard_segment_debug.txt"), "w"
        ) as f:
            f.write(f"max_segments: {max_segments}\n")
            f.write(f"max_segment: {max_segment}\n")
            f.write(f"avg_top5: {avg_top5}\n")
            f.write(f"guard_gauge_ratio_segment: {guard_gauge_ratio_segment}\n")
            f.write(
                f"guard_gauge_ratio_segment_avg5: {guard_gauge_ratio_segment_avg5}\n"
            )
        guard_half_rgb = cv2.cvtColor(guard_half, cv2.COLOR_BGR2RGB)
        guard_ocr = pytesseract.image_to_string(guard_half_rgb)
        return {
            "guard_gauge_colwise": guard_percent_colwise,
            "guard_gauge_ratio_segment": guard_gauge_ratio_segment,
            "guard_gauge_ratio_segment_avg5": guard_gauge_ratio_segment_avg5,
            "guard_gauge_ocr": guard_ocr,
        }

    def extract_ratio(full_img, region, x_offset, y_offset, player_label, debug_dir):
        ratio_width = int(0.12 * full_img.shape[1])
        ratio_height = region.shape[0]
        ratio_y_start = y_offset
        ratio_y_end = min(y_offset + ratio_height, full_img.shape[0])
        if player_label == "player1":
            ratio_x_start = max(x_offset, 0)
            ratio_x_end = min(x_offset + ratio_width, full_img.shape[1])
        else:
            ratio_x_end = full_img.shape[1]
            ratio_x_start = max(ratio_x_end - ratio_width, 0)
        ratio_region = full_img[ratio_y_start:ratio_y_end, ratio_x_start:ratio_x_end]
        if ratio_region.size == 0:
            return ""
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_ratio_region.jpg"), ratio_region
        )
        ratio_hsv = cv2.cvtColor(ratio_region, cv2.COLOR_BGR2HSV)
        lower_purple = np.array([130, 80, 80])
        upper_purple = np.array([155, 255, 255])
        lower_cyan = np.array([85, 80, 80])
        upper_cyan = np.array([105, 255, 255])
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([80, 255, 255])
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
        max_ratio = max(counts, key=counts.get)
        if counts[max_ratio] > 0:
            return max_ratio
        # fallback a OCR si no hay color dominante
        ratio_region_rgb = cv2.cvtColor(ratio_region, cv2.COLOR_BGR2RGB)
        ratio_ocr_raw = pytesseract.image_to_string(
            ratio_region_rgb, config="--psm 8 -c tessedit_char_whitelist=1234"
        )
        import re

        match = re.search(r"[1-4]", ratio_ocr_raw)
        return match.group(0) if match else ""

    def get_bar_info(region, full_img, x_offset, y_offset, player_label):
        # Barras de super (zona inferior)
        if player_label == "player1":
            super_green_pct, super_cyan_pct = analyze_super_bar(
                super1_region, player_label
            )
        else:
            super_green_pct, super_cyan_pct = analyze_super_bar(
                super2_region, player_label
            )
        region_proc = region.copy()
        hsv = cv2.cvtColor(region_proc, cv2.COLOR_BGR2HSV)
        save_debug_image(os.path.join(debug_dir, f"{player_label}_region_hsv.jpg"), hsv)
        mask_yellow, mask_orange, mask_red = segment_health_colors(hsv)
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_mask_yellow.jpg"), mask_yellow
        )
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_mask_orange.jpg"), mask_orange
        )
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_mask_red.jpg"), mask_red
        )
        mask_bar = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, mask_red))
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_mask_bar.jpg"), mask_bar
        )
        health_bar_ratio_segment, health_bar_ratio_segment_avg5 = analyze_health_bar(
            mask_bar, debug_dir, player_label
        )
        # Colwise method (for debug/legacy)
        bar_cols = mask_bar.shape[1]
        bar_rows = mask_bar.shape[0]
        threshold = int(0.5 * bar_rows)
        debug_col_array = np.zeros((bar_rows, bar_cols, 3), dtype=np.uint8)
        col_filled = np.zeros(bar_cols, dtype=bool)
        if player_label == "player1":
            last_filled = -1
            for col in range(bar_cols):
                if np.count_nonzero(mask_bar[:, col]) >= threshold:
                    last_filled = col
                    col_filled[col] = True
                    debug_col_array[:, col] = (0, 255, 0)
                else:
                    debug_col_array[:, col] = (0, 0, 255)
            health_bar_ratio = (
                (last_filled + 1) / bar_cols if bar_cols > 0 and last_filled >= 0 else 0
            )
            if last_filled >= 0:
                cv2.line(
                    debug_col_array,
                    (last_filled, 0),
                    (last_filled, bar_rows - 1),
                    (255, 0, 0),
                    2,
                )
        else:
            first_filled = bar_cols
            for col in range(bar_cols - 1, -1, -1):
                if np.count_nonzero(mask_bar[:, col]) >= threshold:
                    first_filled = col
                    col_filled[col] = True
                    debug_col_array[:, col] = (0, 255, 0)
                else:
                    debug_col_array[:, col] = (0, 0, 255)
            health_bar_ratio = (
                (bar_cols - first_filled) / bar_cols
                if bar_cols > 0 and first_filled < bar_cols
                else 0
            )
            if first_filled < bar_cols:
                cv2.line(
                    debug_col_array,
                    (first_filled, 0),
                    (first_filled, bar_rows - 1),
                    (255, 0, 0),
                    2,
                )
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_mask_bar_cols_filled_color.jpg"),
            debug_col_array,
        )
        debug_col_array_bin = np.where(col_filled, 255, 0).astype(np.uint8)
        debug_col_array_bin = np.tile(debug_col_array_bin, (bar_rows, 1))
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_mask_bar_cols_filled.jpg"),
            debug_col_array_bin,
        )
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
        # Color percentages and health state
        total_bar_pixels = region.shape[0] * region.shape[1]
        yellow_pixels = cv2.countNonZero(mask_yellow)
        orange_pixels = cv2.countNonZero(mask_orange)
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pct = yellow_pixels / total_bar_pixels if total_bar_pixels > 0 else 0
        orange_pct = orange_pixels / total_bar_pixels if total_bar_pixels > 0 else 0
        red_pct = red_pixels / total_bar_pixels if total_bar_pixels > 0 else 0
        if yellow_pct > orange_pct and yellow_pct > red_pct:
            health_state = "high"
        elif orange_pct > red_pct:
            health_state = "medium"
        else:
            health_state = "low"
        with open(
            os.path.join(debug_dir, f"{player_label}_bar_colors_debug.txt"), "w"
        ) as f:
            f.write(f"yellow_pixels: {yellow_pixels}\n")
            f.write(f"orange_pixels: {orange_pixels}\n")
            f.write(f"red_pixels: {red_pixels}\n")
            f.write(f"yellow_pct: {yellow_pct}\n")
            f.write(f"orange_pct: {orange_pct}\n")
            f.write(f"red_pct: {red_pct}\n")
            f.write(f"health_state: {health_state}\n")
        # Name and guard
        name_region, name_half, guard_half = extract_name_and_guard_regions(
            full_img, region, x_offset, y_offset, player_label
        )
        if name_region.size > 0:
            save_debug_image(
                os.path.join(debug_dir, f"{player_label}_name_region.jpg"), name_region
            )
        if name_half.size > 0:
            save_debug_image(
                os.path.join(debug_dir, f"{player_label}_name_half.jpg"), name_half
            )
        if guard_half.size > 0:
            save_debug_image(
                os.path.join(debug_dir, f"{player_label}_guard_region.jpg"), guard_half
            )
        if name_half.size > 0:
            name_half_rgb = cv2.cvtColor(name_half, cv2.COLOR_BGR2RGB)
            name_ocr = pytesseract.image_to_string(name_half_rgb)
        else:
            name_ocr = ""
        guard_gauge_info = extract_guard_gauge(guard_half, player_label, debug_dir)
        # Ratio
        ratio_detected = extract_ratio(
            full_img, region, x_offset, y_offset, player_label, debug_dir
        )
        # OCR de la barra (por si hay texto relevante)
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        ocr_text = pytesseract.image_to_string(region_rgb)
        save_debug_image(
            os.path.join(debug_dir, f"{player_label}_region_rgb.jpg"), region
        )
        return {
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
            "guard_gauge_colwise": (
                guard_gauge_info["guard_gauge_colwise"] if guard_gauge_info else None
            ),
            "guard_gauge_ratio_segment": (
                guard_gauge_info["guard_gauge_ratio_segment"]
                if guard_gauge_info
                else None
            ),
            "guard_gauge_ratio_segment_avg5": (
                guard_gauge_info["guard_gauge_ratio_segment_avg5"]
                if guard_gauge_info
                else None
            ),
            "guard_gauge_ocr": (
                guard_gauge_info["guard_gauge_ocr"].strip() if guard_gauge_info else ""
            ),
            "ratio": ratio_detected.strip(),
            "super_bar_green_pct": super_green_pct,
            "super_bar_cyan_pct": super_cyan_pct,
        }

    player1_info = get_bar_info(player1_region, img, 0, 0, "player1")
    player2_info = get_bar_info(player2_region, img, int(0.55 * width), 0, "player2")

    output = {
        "player1": {
            **player1_info,
            "region_analyzed": {
                "x_min": 0,
                "y_min": 0,
                "x_max": int(0.45 * width),
                "y_max": bar_height,
            },
        },
        "player2": {
            **player2_info,
            "region_analyzed": {
                "x_min": int(0.55 * width),
                "y_min": 0,
                "x_max": width,
                "y_max": bar_height,
            },
        },
        "round_timer": round_timer,
    }

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
    health: str = Field(
        ...,
        description="Health bar for the player, usually in a solid color which can be yellow (full or almost full) to red (dangerously low).",
    )
    ratio: int = Field(
        ..., ge=1, le=4, description="Ratio number for the current character (1-4)."
    )
    guard_gauge: str = Field(
        ...,
        description="Small bar below the health bar that shows the guard gauge status for the player.",
    )
    position: List[int] = Field(
        ...,
        description="Bounding box of the player in the format [x_min, y_min, x_max, y_max] in image coordinates.",
    )


class FightingGameHUD(BaseModel):
    player1: PlayerHUD = Field(
        ..., description="HUD information for the player on the left (Player 1)."
    )
    player2: PlayerHUD = Field(
        ..., description="HUD information for the player on the right (Player 2)."
    )
    round_timer: int = Field(
        ...,
        ge=0,
        le=999,
        description="This timer is in the upper middle of the screen and shows how much time is left in the round.",
    )
    combo_counter_messages: str = Field(
        ...,
        description="The numbers show how many hits you have done in a combo. Below it, there can also appear messages like 'Guard Crush', 'Reversal', 'Counter', and the type of K.O. in the end of a round.",
    )
