import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Carregar o dataset
dataset_patch = os.path.join('..', 'dataset_new')
df_atual_df = os.path.join(dataset_patch, '7-Perfil_Compostagem(meio).csv')
df = pd.read_csv(df_atual_df)

# Renomear as colunas para facilitar a leitura
df = df.rename(columns={
    'D0': 'VOC (ppb)', 
    'D1': 'VOC (ug m-3)', 
    'D2': 'VOC (Temperatura)', 
    'D3': 'VOC (Umidade)',
    'D4': 'H2S (ppb)', 
    'D5': 'H2S (ug m-3)', 
    'D6': 'H2S (Temperatura)', 
    'D7': 'H2S (Umidade)',
    'D8': 'SO2 (ppb)', 
    'D9': 'SO2 (ug m-3)', 
    'D10': 'SO2 (Temperatura)', 
    'D11': 'SO2 (Umidade)',
    'D12': 'NH3 (ppb)', 
    'D13': 'NH3 (ug m-3)', 
    'D14': 'NH3 (Temperatura)', 
    'D15': 'NH3 (Umidade)',
    'D16': 'CH3SH (ppb)', 
    'D17': 'CH3SH (ug m-3)', 
    'D18': 'CH3SH (Temperatura)', 
    'D19': 'CH3SH (Umidade)',
    'D26': 'BME280 - TEMPERATURA', 
    'D27': 'BME280 - PRESSAO', 
    'D28': 'BME280 - UMIDADE'
})

# Definir a coluna a ser analisada
coluna_analise = 'VOC (ppb)'

# Calcular a diferença absoluta entre os pontos consecutivos
df['diff'] = df[coluna_analise].diff().abs()

# Definir um valor de threshold para detectar estabilização (ajustar conforme necessário)
threshold = 0.4 # Ajuste este valor dependendo do nível de estabilização desejado

# Encontrar o índice onde a diferença consecutiva fica consistentemente abaixo do threshold
window_size = 50  # Tamanho da janela para considerar estabilização
df['stable'] = df['diff'].rolling(window_size).apply(lambda x: (x < threshold).all(), raw=True)

# Identificar o ponto de estabilização
stabilization_index = df[df['stable'] == 1].index[0] if df['stable'].any() else 0

# Remover o período de estabilização
df_stabilized = df.loc[stabilization_index:].reset_index(drop=True)

# Plotar o resultado
plt.plot(df_stabilized[coluna_analise])
plt.title(f'{coluna_analise} após tempo de estabilização')
plt.ylabel(coluna_analise)
plt.xlabel('Tempo')
plt.show()

# Exibir o índice de estabilização encontrado
print(f'Tempo de estabilização identificado no índice: {stabilization_index}')
