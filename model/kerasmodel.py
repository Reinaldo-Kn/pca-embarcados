import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tcn import TCN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt

# Load dataset
dataset_patch = os.path.join('..', 'dataset_new')
df_atual_df = os.path.join(dataset_patch, 'dirty-concatV4.csv')

df = pd.read_csv(df_atual_df)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#df = df.drop(['D20', 'D21', 'D22', 'D23', 'D24', 'D25'], axis=1)
df = df.dropna()

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

label_encoder = LabelEncoder()


y = label_encoder.fit_transform(y)
print(label_encoder.classes_)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Reshape input for TCN
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
# Define the input layer
input_layer = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))

# Add the TCN layers
tcn_layer = TCN(64, return_sequences=False)(input_layer)  

output_layer = keras.layers.Dense(len(np.unique(y)), activation='softmax')(tcn_layer)

model = keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=-1)

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

# Optionally, save the model
model.save(os.path.join(dataset_patch, 'tcn_model-211.h5'))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()
