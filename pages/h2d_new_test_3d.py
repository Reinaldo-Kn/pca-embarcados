import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from dash import dcc, html
import dash
from dash_bootstrap_templates import load_figure_template
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Configurações do Dash
dash.register_page(__name__, path="/test3D", name="Teste Dataset 3D")
load_figure_template(["yeti", "yeti_dark"])

# Carregar dados
df_main = pd.read_csv('./dataset_new/concatV4-names.csv')
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

# Padronização dos dados
x_balanced = StandardScaler().fit_transform(x_balanced)

# Aplicação de PCA
pca_odor = PCA(n_components=3)
principalComponents_odor = pca_odor.fit_transform(x_balanced)
total_var = pca_odor.explained_variance_ratio_.sum() * 100

components = principalComponents_odor

# Plotagem
fig = px.scatter_3d(
    x=principalComponents_odor[:, 0], 
    y=principalComponents_odor[:, 1], 
    z=principalComponents_odor[:, 2], 
    color=y_balanced,
    labels={'x': 'Componente 1', 'y': 'Componente 2', 'z': 'Componente 3'},
    title=f'Análise PCA 3D - Variância Total Explicada: {total_var:.2f}%'
)

# Layout do Dash
layout = html.Div([
    html.H1("Análise de Redução Dimensional 3D"),
    dcc.Graph(figure=fig)
])