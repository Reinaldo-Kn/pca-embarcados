import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import pandas as pd
import umap

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
y_balanced = df_odor_balanced['Class'].values

# Aplicar UMAP para redução de dimensionalidade
umap_model = umap.UMAP(n_components=2, 
                       random_state=42,
                       n_epochs=100,
                       n_neighbors=500,
                        spread=2,
                        min_dist=0.5,
                       metric='euclidean')
X_umap = umap_model.fit_transform(x_balanced)

# Plotar os resultados do UMAP
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_balanced, palette='Set2', s=100, edgecolor='black')
plt.title("UMAP - Primeiros Dois Componentes", fontsize=15)
plt.xlabel('Componente 1', fontsize=12)
plt.ylabel('Componente 2', fontsize=12)
plt.legend(title='Classe', title_fontsize=12, fontsize=10)
plt.show()
