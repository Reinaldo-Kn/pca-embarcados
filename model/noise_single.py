import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


dataset_patch = os.path.join('..', 'dataset_new')
df_atual_df = os.path.join(dataset_patch, 'Perfil-Geral-Alho-v2.csv')
df = pd.read_csv(df_atual_df)

time_column = df.iloc[:, 0].copy()


X = df.iloc[:, 1:-1].copy()  
y = df.iloc[:, -1].copy()    

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Quantidade de dados antes da adição de ruído:", X.shape[0])

noise_factor = 0.05  
X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)


X_combined = np.vstack((X, X_noisy))
y_combined = np.hstack((y_encoded, y_encoded))

print("Quantidade de dados após a adição de ruído:", X_combined.shape[0])

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

df_resampled = pd.DataFrame(X_resampled, columns=X.columns)

y_resampled_labels = label_encoder.inverse_transform(y_resampled)

df_resampled['label'] = y_resampled_labels
df_resampled['TIME'] = np.resize(time_column.values, df_resampled.shape[0])

columns_order = ['TIME'] + [col for col in df_resampled.columns if col != 'TIME' and col != 'label'] + ['label']
df_resampled = df_resampled[columns_order]

df_resampled.to_csv(os.path.join(dataset_patch, 'Perfil-Geral-Alho-v2_resample.csv'), index=False)

print("Quantidade de dados após a aplicação do SMOTE:", df_resampled.shape[0])
