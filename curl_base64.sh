#!/bin/bash
# Uso: ./curl_base64.sh imagen.jpg [URL]
# Si no se especifica URL, usa http://localhost:8000/generate_base64

IMG="$1"
URL="${2:-http://localhost:8000/generate_base64}"

if [ -z "$IMG" ]; then
  echo "Uso: $0 imagen.jpg [URL]"
  exit 1
fi

# Redimensionar la imagen a 1280x720 (720p) y guardar como temporal
TMP_IMG="resized_$$.jpg"
convert "$IMG" -resize 1280x720\! "$TMP_IMG"

BASE64_IMG=$(base64 -w 0 "$TMP_IMG")
echo "{\"image_base64\": \"${BASE64_IMG}\"}" > payload.json

curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d @payload.json

rm payload.json "$TMP_IMG"
