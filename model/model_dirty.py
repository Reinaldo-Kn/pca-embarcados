import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib , os
import matplotlib.pyplot as plt
import numpy as np  


dataset_patch = os.path.join('..','dataset_new')
df_atual_df = os.path.join(dataset_patch,'dirty-concatV4.csv')
df= pd.read_csv(df_atual_df)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna()
features = df.columns[1:-1]
df_features = df.iloc[:, 1:-1].copy()
df_labels = df.iloc[:,-1].copy()

X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42, stratify=df_labels)

# 4. Treinar um modelo de classificação com validação cruzada
model = RandomForestClassifier(random_state=42,class_weight='balanced')

# Definir os hiperparâmetros para otimização
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5,],
    'min_samples_leaf': [2, 4]
}

# Utilizar GridSearchCV para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
best_model = grid_search.best_estimator_

# 5. Avaliar o modelo
y_pred = best_model.predict(X_test)

# 6. Imprimir a matriz de confusão e o relatório de classificação
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Salvar o modelo para uso futuro
joblib.dump(best_model, 'modelo_classificacao_dirty-2.pkl')