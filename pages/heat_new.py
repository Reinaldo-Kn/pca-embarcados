import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from dash_bootstrap_templates import load_figure_template
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
# Initialize the app
dash.register_page(__name__, path="/heat", name="Novo PCA Variação")

load_figure_template(["yeti","yeti_dark"])

# Load your dataset (replace 'concat.csv' with your actual path)
df_main = pd.read_csv('./dataset_lab/concatV6_40.csv')

# Select features and labels
features = df_main.columns[1:-1]
df_features = df_main.iloc[:, 2:-1].copy()
df_labels = df_main.iloc[:,-1].copy()

# Layout of the app
layout = html.Div([
    html.H4("Visualization of PCA's explained variance"),
    dcc.Graph(id="graph"),
    html.P("Number of components:"),
    dcc.Slider(
        id='slider',
        min=2, max=min(len(features), 5),  # Adjust max to your dataset's feature count
        value=3, step=1
    )
])

# Callback to update the graph based on the number of components
@dash.callback(
    Output("graph", "figure"), 
    Input("slider", "value"))
def run_and_plot(n_components):
    # PCA computation
    
    #use standard scaler
    
    x = StandardScaler().fit_transform(df_features)
    
    pca_odor = PCA(n_components=n_components)
    principalComponents_odor = pca_odor.fit_transform(x)

    # Explained variance
    total_var = pca_odor.explained_variance_ratio_.sum() * 100
    loadings = pca_odor.components_.T * np.sqrt(pca_odor.explained_variance_)

    # Labels for axes
    labels = {str(i): f"PC {i+1}" for i in range(n_components)}
    labels['color'] = 'Class'

    # Create the scatter plot using Plotly Express
    fig = px.scatter_matrix(
        pd.DataFrame(principalComponents_odor),
        dimensions=range(n_components),
        color=df_labels,
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}%'
    )
    fig.update_traces(diagonal_visible=False)
    
    return fig
