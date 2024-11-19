import random
import time
import requests
import json
from flask import Flask, request, jsonify
from threading import Thread

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    print(f"Received data: {data}")
    return jsonify({'status': 'success'}), 200


def fake_api():
    url = 'http://' + 'localhost' + ':' + '9000' + '/api/data'
    sensors = ['D' + str(i) for i in range(35)]  # Sensores D0 a D34
    
    headers = {'Content-Type': 'application/json'}
    
    while True:
        try:
            data = {sensor: random.random() for sensor in sensors}
            
            response = requests.post(url, data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                print(f"Data sent successfully: {data}")
            else:
                print(f"Failed to send data. Status code: {response.status_code}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
        
        time.sleep(2)  # Envia dados a cada 2 segundos


if __name__ == '__main__':
    # Rodar o servidor Flask em uma thread separada
    server_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=9000))
    server_thread.daemon = True  # Permite que a thread seja encerrada quando o programa principal finalizar
    server_thread.start()
    
    # Executar o envio de dados da API falsa
    fake_api()
