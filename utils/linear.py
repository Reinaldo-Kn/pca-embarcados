import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import pandas as pd

df_main = pd.read_csv('../dataset_new/concatV4-names.csv')
df_main = df_main.drop(['TIME'], axis=1)
features = df_main.columns[:-1]
df_features = df_main[features].copy()
df_labels = df_main.iloc[:, -1].copy()

# Concatenar features e labels para facilitar o balanceamento
df_odor = pd.concat([df_features, df_labels], axis=1)

# Separar as classes
compostagem_final = df_odor[df_odor['Class'] == 'Compostagem(final)']
compostagem_meio = df_odor[df_odor['Class'] == 'Compostagem(meio)']
compostagem_inicio = df_odor[df_odor['Class'] == 'Compostagem(inicio)']
racao = df_odor[df_odor['Class'] == 'Racao']
pau_de_alho = df_odor[df_odor['Class'] == 'Pau de alho']

# Subamostragem para igualar o tamanho das classes
compostagem_final_downsampled = resample(compostagem_final, replace=False, n_samples=1000, random_state=82)
compostagem_meio_downsampled = resample(compostagem_meio, replace=False, n_samples=1000, random_state=84)
compostagem_inicio_downsampled = resample(compostagem_inicio, replace=False, n_samples=1000, random_state=86)
racao_downsampled = resample(racao, replace=False, n_samples=1000, random_state=42)
pau_de_alho_downsampled = resample(pau_de_alho, replace=False, n_samples=1000, random_state=52)

# Reunir as classes balanceadas
df_odor_balanced = pd.concat([
    compostagem_final_downsampled,
    compostagem_meio_downsampled,
    compostagem_inicio_downsampled,
    racao_downsampled,
    pau_de_alho_downsampled
])

# Separar novamente em features e labels
x_balanced = df_odor_balanced[features].values


#matriz de correlação
correlation_matrix = np.corrcoef(x_balanced, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Matriz de Correlação")
plt.show()

fraca_limite = 0.3  # Correlações entre -0.3 e 0.3 são consideradas fracas
forte_limite = 0.7  # Correlações acima de 0.7 (ou abaixo de -0.7) são fortes

# Retirando a diagonal (correlação de cada variável consigo mesma, que é 1)
mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
correlation_values = correlation_matrix[mask]

# Contagem de correlações fortes, fracas, positivas e negativas
fraca = np.sum((correlation_values > -fraca_limite) & (correlation_values < fraca_limite))
forte = np.sum((correlation_values >= forte_limite) | (correlation_values <= -forte_limite))
positiva = np.sum(correlation_values > 0)
negativa = np.sum(correlation_values < 0)

# Exibindo os resultados no terminal
print(f"Quantidade de correlações fracas: {fraca}")
print(f"Quantidade de correlações fortes: {forte}")
print(f"Quantidade de correlações positivas: {positiva}")
print(f"Quantidade de correlações negativas: {negativa}")