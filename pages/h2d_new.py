import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from dash import dcc, html
import dash
from dash_bootstrap_templates import load_figure_template
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np
# Configurações do Dash
dash.register_page(__name__, path="/fig2dNEW", name="LAB PCA em 2D")
load_figure_template(["yeti", "yeti_dark"])

# Carregar dados
df_main = pd.read_csv('./dataset_lab/concatV6_40.csv')
df_main = df_main.drop(['TIME'], axis=1)
features = df_main.columns[:-1]
df_features = df_main[features].copy()
df_labels = df_main.iloc[:, -1].copy()

# Concatenar features e labels para facilitar o balanceamento
df_odor = pd.concat([df_features, df_labels], axis=1)



# print("Distribuição das classes após o balanceamento:")
# print(df_odor_balanced['Class'].value_counts())


# Separar novamente em features e labels
x_balanced = df_odor[features].values
y_balanced = df_odor['Class'].values

# Padronização dos dados
x_balanced = StandardScaler().fit_transform(x_balanced)


pca_odor = PCA(n_components=2)
principalComponents_odor = pca_odor.fit_transform(x_balanced)
total_var = pca_odor.explained_variance_ratio_.sum() * 100

components = principalComponents_odor

loadings = pca_odor.components_.T * np.sqrt(pca_odor.explained_variance_)



scale_factor =  4 # Ajuste este valor para aumentar ou diminuir a distância entre as setas

# Aplique o fator de escala aos loadings
loadings = loadings * scale_factor

total_var = pca_odor.explained_variance_ratio_.sum() * 100


# Plotagem
fig = px.scatter(
    x=components[:, 0], 
    y=components[:, 1], 
    color=y_balanced,
    title=f'PCA - Variância total: {total_var:.2f}%',
    labels={'x': 'Componente 1', 'y': 'Componente 2'}
)
x_shifts = [10 * ((-1)**i) for i in range(len(features))]  # Alterna para a direita/esquerda
y_shifts = [10 * ((-1)**(i + 1)) for i in range(len(features))]  
for i, feature in enumerate(features):
    fig.add_annotation(
        ax=0, ay=0,
        axref="x", ayref="y",
        x=loadings[i, 0],
        y=loadings[i, 1],
        showarrow=True,
        arrowsize=2,
        arrowhead=2,
        xanchor="right",
        yanchor="top"
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        xshift=x_shifts[i],
        yshift=y_shifts[i],
    )

# Layout do Dash
layout = html.Div([
    html.H1("Análise de Redução Dimensional "),
    dcc.Graph(figure=fig)
])