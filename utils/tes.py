import requests
import random
import time

url = 'http://localhost:9001/api/data'

# Função para gerar dados de sensores falsos
def generate_sensor_data():
    return {'D' + str(i): random.randint(0, 100) for i in range(35)}

# Loop para enviar dados a cada 5 segundos
while True:
    data = generate_sensor_data()
    response = requests.post(url, json=data)
    print(f"Enviando dados: {data}, Status: {response.status_code}")
    time.sleep(5)
