import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carregar o dataset
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
y_balanced = df_odor_balanced['Class'].values  # Rótulos das classes

# Passo 1: Normalizar os dados
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_balanced)  # Normaliza as features

# Passo 2: Aplicar PCA
pca = PCA()
x_pca = pca.fit_transform(x_scaled)

# Passo 3: Variância explicada pelos componentes principais
explained_variance = pca.explained_variance_ratio_

# Plotar a variância explicada por cada componente principal
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Componentes Principais')
plt.ylabel('Variância Explicada')
plt.title('Variância Explicada por Cada Componente Principal')
plt.show()

# Passo 4: Variância acumulada
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Acumulada')
plt.title('Variância Acumulada pelos Componentes Principais')
plt.grid(True)
plt.show()

# Definir o número de componentes com base na variância acumulada (por exemplo, 95%)
threshold = 0.95  # Manter 95% da variância
n_components = np.argmax(cumulative_variance >= threshold) + 1

# Exibir o número de componentes escolhidos
print(f"Número de componentes principais que explicam 95% da variância: {n_components}")

# Passo 5: Aplicar PCA com o número de componentes selecionados
pca = PCA(n_components=n_components)
x_pca_reduced = pca.fit_transform(x_scaled)

# Exibir a forma dos dados após a redução de dimensionalidade
print(f"Forma dos dados após PCA: {x_pca_reduced.shape}")

# Passo 6: Gráfico 2D dos primeiros dois componentes principais
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x_pca_reduced[:, 0], y=x_pca_reduced[:, 1], hue=y_balanced, palette="Set1", s=60, alpha=0.7)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Primeiros Dois Componentes Principais')
plt.legend(title='Classe', loc='best')
plt.show()
