import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Carregar dados
df_main = pd.read_csv('../dataset_new/concatV5.csv')
df_main = df_main.drop(['TIME'], axis=1)
features = df_main.columns[:-1]
df_features = df_main[features].copy()
df_labels = df_main.iloc[:, -1].copy()

# Concatenar features e labels para facilitar o balanceamento
df_odor = pd.concat([df_features, df_labels], axis=1)

# Separar as classes (sem subamostragem)
compostagem = df_odor[df_odor['Class'] == 'Compostagem']
racao = df_odor[df_odor['Class'] == 'Racao']
pau_de_alho = df_odor[df_odor['Class'] == 'Pau de alho']

# Concatenar as classes diretamente
df_odor_balanced = pd.concat([compostagem, racao, pau_de_alho])

# Separar novamente em features e labels
x_balanced = df_odor_balanced[features].values
y_balanced = df_odor_balanced['Class'].values

# Padronização dos dados
# x_balanced = StandardScaler().fit_transform(x_balanced)

# Definindo a função de pontuação para o GridSearch
class UMAPWithSilhouetteScore(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=15, min_dist=0.1, n_epochs=200):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs

    def fit(self, X, y=None):
        self.umap_ = umap.UMAP(n_neighbors=self.n_neighbors, 
                               min_dist=self.min_dist, 
                               n_epochs=self.n_epochs, 
                               random_state=42)
        self.umap_.fit(X)
        return self

    def transform(self, X):
        return self.umap_.transform(X)

    def score(self, X, y=None):
        # Calcula o silhouette score para a transformação
        umap_components = self.transform(X)
        score = silhouette_score(X, umap_components)
        return score

# Definindo os parâmetros do GridSearch
param_grid = {
    'n_neighbors': [5, 10, 15],
    'min_dist': [0.1, 0.3, 0.5],
    'n_epochs': [200, 500]
}

# Inicializando o GridSearchCV
umap_model = UMAPWithSilhouetteScore()
grid_search = GridSearchCV(umap_model, param_grid, scoring='neg_mean_squared_error', cv=3)

# Ajustando o modelo
grid_search.fit(x_balanced)

# Exibindo os melhores parâmetros
print(f'Melhores parâmetros encontrados: {grid_search.best_params_}')
print(f'Score de Silhueta: {-grid_search.best_score_}')

# UMAP com os melhores parâmetros encontrados
best_umap = grid_search.best_estimator_
best_umap_components = best_umap.transform(x_balanced)

# Plotagem com os melhores parâmetros encontrados
plt.figure(figsize=(8, 6))
plt.scatter(best_umap_components[:, 0], best_umap_components[:, 1], c=y_balanced, cmap='viridis')
plt.colorbar(label='Class')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title(f'UMAP - Melhor combinação de parâmetros: n_neighbors={grid_search.best_params_["n_neighbors"]}, '
          f'min_dist={grid_search.best_params_["min_dist"]}, n_epochs={grid_search.best_params_["n_epochs"]}\n'
          f'Score de Silhueta: {-grid_search.best_score_:.4f}')
plt.show()
