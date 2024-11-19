import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tcn import TCN 

# Load the dataset
dataset_patch = os.path.join('..', 'dataset_new')
df_atual_df = os.path.join(dataset_patch, '1-Coleta_04-11-24 - MEDIDAS1.csv')
df = pd.read_csv(df_atual_df)

# Preprocessing
df = df.drop(['TIME'], axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(['D20', 'D21', 'D22', 'D23', 'D24', 'D25'], axis=1)
df = df.dropna()


model_patch = os.path.join('..', 'dataset_new','tcn_model-2.h5')
with tf.keras.utils.custom_object_scope({'TCN': TCN}):  # Register the TCN layer
    model = tf.keras.models.load_model(model_patch) 

X_new = df.values  # Get the feature values
X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))  # Reshape for TCN input

# Make predictions
y_pred = model.predict(X_new)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability




class_mapping = {0: 'Compostagem (final)', 1: 'Compostagem (inicio)',2:'Compostagem (meio)',3:'Pau de alho',4:'Racao'}

# Count occurrences and percentages of each class
value_counts = pd.Series(y_pred_classes).value_counts()
total = len(y_pred_classes)
for class_label, count in value_counts.items():
    class_name = class_mapping.get(class_label, "Desconhecido")
    percentage = (count / total) * 100
    print(f'Classe {class_name}: {percentage:.2f}% ({count} ocorrências)')

# Map predicted classes to labels and print the mapped occurrences
y_pred_mapped = pd.Series(y_pred_classes).map(class_mapping)

print("\nOcorrências por classe:")
print(y_pred_mapped.value_counts())