# import plotly.express as px
# import pandas as pd
# from sklearn.decomposition import PCA
# from dash import dcc, html
# import dash
# from dash_bootstrap_templates import load_figure_template
# from sklearn.preprocessing import StandardScaler ,MinMaxScaler , RobustScaler, PowerTransformer
# from sklearn.utils import resample
# from sklearn.decomposition import KernelPCA


# # Configurações do Dash
# dash.register_page(__name__, path="/testkernal", name="kernal Dataset")
# load_figure_template(["yeti", "yeti_dark"])

# # Carregar dados
# df_main = pd.read_csv('./dataset_lab/concatV6.csv')
# df_main = df_main.drop(['TIME'], axis=1)
# features = df_main.columns[:-1]
# df_features = df_main[features].copy()
# df_labels = df_main.iloc[:, -1].copy()

# # Concatenar features e labels para facilitar o balanceamento
# df_odor = pd.concat([df_features, df_labels], axis=1)


# # Separar novamente em features e labels
# x_balanced = df_odor[features].values
# y_balanced = df_odor['Class'].values

# # Padronização dos dados
# x_scaled = PowerTransformer().fit_transform(x_balanced)

# # Aplicação de PCA


# kpca = KernelPCA(n_components=2, kernel='rbf')  # Ajuste o valor de gamma
# X_kpca = kpca.fit_transform(x_scaled)

# # pca_odor = PCA(n_components=3)
# # principalComponents_odor = pca_odor.fit_transform(x_scaled)
# #total_var = pca_odor.explained_variance_ratio_.sum() * 100

# # components = principalComponents_odor

# # Plotagem
# fig = px.scatter(
#     x=X_kpca[:, 0], 
#     y=X_kpca[:, 1], 
#     color=y_balanced,
#     labels={'x': 'Componente 1', 'y': 'Componente 2'},
#     #title=f"Análise PCA - Variância Total Explicada: {total_var:.2f}%"
# )

# # Layout do Dash
# layout = html.Div([
#     html.H1("Análise de Redução Dimensional KERNAL PCA "),
#     dcc.Graph(figure=fig)
# ])
