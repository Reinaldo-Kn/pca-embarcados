import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from dash import dcc, html
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# Inicialize o app
dash.register_page(__name__, path="/fig2d", name="PCA em 2D")

load_figure_template(["yeti", "yeti_dark"])

# Carregar e preparar os dados
df = pd.read_csv('./dataset/All_adjust.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

features = df.columns[:-1]
features_data = df[features]

features_label = df.iloc[:, -1]
odor_dataset = pd.concat([features_data, features_label], axis=1)

# Padronizar os dados
x = odor_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)

# PCA para reduzir a dimensionalidade para 2D
pca_odor = PCA(n_components=6)
principalComponents_odor = pca_odor.fit_transform(x)

# Aplicar DBSCAN para detecção de outliers
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Ajuste os parâmetros conforme necessário
dbscan_labels = dbscan.fit_predict(principalComponents_odor)

# Atribuir um rótulo para os outliers (-1 são os outliers)
odor_dataset['DBSCAN_Label'] = dbscan_labels

# PCA Loadings
loadings = pca_odor.components_.T * np.sqrt(pca_odor.explained_variance_)

total_var = pca_odor.explained_variance_ratio_.sum() * 100

# Plotar gráfico de dispersão colorido pelas classes
fig = px.scatter(
    x=principalComponents_odor[:, 0], 
    y=principalComponents_odor[:, 1], 
    color=odor_dataset['Class'],  # Colorir pelos valores da coluna 'Class'
    title=f'PCA com DBSCAN - Variância total: {total_var:.2f}%',
)

# Adicionar outliers detectados pelo DBSCAN em rosa choque
outliers = principalComponents_odor[odor_dataset['DBSCAN_Label'] == -1]
fig.add_scatter(
    x=outliers[:, 0],
    y=outliers[:, 1],
    mode='markers',
    marker=dict(color='hotpink', size=10, symbol='x'),
    name='DBSCAN Outliers'
)

# Adicionar setas das variáveis
for i, feature in enumerate(features):
    fig.add_annotation(
        ax=0, ay=0,
        axref='x', ayref='y',
        x=loadings[i, 0],
        y=loadings[i, 1],
        showarrow=True,
        arrowsize=2,
        arrowhead=2,
        xanchor='right',
        yanchor='top',
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        yshift=5,
    )

# Layout da página
layout = html.Div([
    html.H1("Análise PCA em 2D com DBSCAN"),
    dcc.Graph(figure=fig)  # Insere o gráfico no layout
])
