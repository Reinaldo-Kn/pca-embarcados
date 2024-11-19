import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import plotly.express as px
from sklearn.decomposition import PCA

# Carregar dados
df_main = pd.read_csv('../dataset_lab/concatV6.csv')
df_main = df_main.dropna()

# Parâmetros do filtro
fs = 100000.0          # Frequência de amostragem do seu sinal (Hz)
cutoff = 2000.0       # Frequência de corte do filtro passa-baixa (Hz)
order = 4              # Ordem do filtro

# Função para criar o filtro Butterworth passa-baixa
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Função para aplicar o filtro
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Aplicar o filtro em todas as colunas de odor (exceto TIME e CLASS)
df_filtered = df_main.copy()
for col in df_filtered.columns[1:-1]:  # Ignorar a primeira (TIME) e última coluna (CLASS)
    df_filtered[col] = lowpass_filter(df_filtered[col], cutoff, fs, order)

# Função para remover 20% do início de cada classe
def remove_20_percent_start(df, class_name):
    class_data = df[df['Class'] == class_name]
    cutoff_index = int(len(class_data) * 0.4)  # Índice de corte em 20%
    return class_data.iloc[cutoff_index:]

# Aplicar a função para cada classe
compostagem_final = remove_20_percent_start(df_filtered,'Compostagem(final)')
compostagem_meio = remove_20_percent_start(df_filtered,'Compostagem(meio)')
compostagem_inicio = remove_20_percent_start(df_filtered,'Compostagem(inicio)')
racao = remove_20_percent_start(df_filtered, 'Racao(frango)')
pau_de_alho = remove_20_percent_start(df_filtered, 'Pau de alho(flor)')

# Concatenar todas as classes novamente
df_odor_filtered = pd.concat([compostagem_final,compostagem_meio,compostagem_inicio, racao, pau_de_alho], axis=0)

# Salvar o novo dataset com 20% removido do início de cada classe
df_odor_filtered.to_csv('../dataset_lab/concatV6_40.csv', index=False)

