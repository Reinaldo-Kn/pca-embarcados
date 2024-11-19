from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

# Verifica se o arquivo CSV já existe, caso contrário, cria-o e adiciona o cabeçalho
csv_file = 'sensor_data.csv'
file_exists = os.path.isfile(csv_file)

# Definir o cabeçalho (nomes dos sensores)
sensors = ['D' + str(i) for i in range(35)]

# Se o arquivo CSV não existe, cria-o com o cabeçalho
if not file_exists:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=sensors)
        writer.writeheader()

# Função para receber dados e salvar no CSV
@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    
    # Salva os dados no arquivo CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=sensors)
        writer.writerow(data)
    
    print(f"Data saved: {data}")
    return jsonify({'status': 'success'}), 200

# Mantém o servidor rodando e escutando
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001)
