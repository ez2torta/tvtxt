import time
import requests

url = "http://localhost:8000/generate"
payload = {
    "image_url": "https://alonsoastroza.com/projects/ft-hackathon/avello.jpg"
}

start = time.time()
response = requests.post(url, json=payload)
elapsed = time.time() - start
print(response.json())
print(f"Tiempo transcurrido: {elapsed:.2f} segundos")