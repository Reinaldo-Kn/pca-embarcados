import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dash import dcc, html
import dash


dash.register_page(__name__,path="/fig3d",name="PCA em 3D")




df = pd.read_csv('./dataset/All_adjust.csv')
#if the column is unnamed, it will be removed
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#padronizando os dados
features = df.columns[:-1]
features_data = df[features]

features_label = df.iloc[:,-1]
odor_dataset = pd.concat([features_data, features_label], axis=1)

x = odor_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)

pca_odor = PCA(n_components=3)
principalComponents_odor = pca_odor.fit_transform(x)

total_var = pca_odor.explained_variance_ratio_.sum() * 100  

fig = px.scatter_3d(
    x=principalComponents_odor[:,0], 
    y=principalComponents_odor[:,1],
    z=principalComponents_odor[:,2],
    color=odor_dataset['Class'],
    title=f'PCA Variância total: {total_var:.2f}%',
    labels={'0': 'Principal Component 1', '1': 'Principal Component 2', '2': 'Principal Component 3'}
)

layout = html.Div([
    html.H1("Análise PCA em 3D"),
    dcc.Graph(figure=fig)  # Insere o gráfico no layout
])
