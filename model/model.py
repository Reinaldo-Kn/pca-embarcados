import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib , os
import matplotlib.pyplot as plt
import numpy as np  
dataset_patch = os.path.join('..','dataset_new')
df_atual_df = os.path.join(dataset_patch,'pca_transformed_noisy_smote.csv')
df= pd.read_csv(df_atual_df)


# 2. Separe as características e o rótulo
X = df[['PC1', 'PC2']]
y = df['label']

# 3. Divida o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treine um modelo de classificação com validação cruzada
model = RandomForestClassifier(random_state=42)

# Defina os hiperparâmetros para otimização
param_grid = {
    'n_estimators': [ 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [ 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced']
}


# Utilize GridSearchCV para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
best_model = grid_search.best_estimator_
# 5. Avalie o modelo
y_pred = best_model.predict(X_test)

# 6. Imprima a matriz de confusão e o relatório de classificação
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Salve o modelo para uso futuro
joblib.dump(best_model, 'modelo_classificacao_new.pkl')


