import pandas as pd
from sklearn.decomposition import PCA
import joblib , os


dataset_patch = os.path.join('..','dataset_new')
df_atual_df = os.path.join(dataset_patch,'1-Coleta_04-11-24 - MEDIDAS1.csv')
# df_pca_original = os.path.join(dataset_patch,'pca_transformed.csv')
df= pd.read_csv(df_atual_df)

#preprocessamento
df = df.drop(['TIME'], axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(['D20','D21','D22','D23','D24','D25'], axis=1)
df = df.dropna()



# Preveja as classes
model = joblib.load('modelo_classificacao_dirty-2.pkl')
y_pred = model.predict(df)
# printa a classe predominante
# sum the total of each class, print the porcentage of each class in the dataset




value_counts = pd.Series(y_pred).value_counts()
total = len(y_pred)
for class_label, count in value_counts.items():
    percentage = (count / total) * 100
    print(f'Classe {class_label}: {percentage:.2f}% ({count} ocorrências)')
    
print(pd.Series(y_pred).value_counts())