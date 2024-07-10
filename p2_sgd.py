import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

x,y = make_classification(n_samples=1000, n_classes=2, n_features=20, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(x_train.shape[1], activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gd_hist = model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=2, verbose=0)


sgd_model = tf.keras.Sequential([
    tf.keras.layers.Dense(x_train.shape[1], activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

sgd_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
sgd_hist = sgd_model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=35, verbose=0)


plt.plot(gd_hist.history['accuracy'], label='training accuracy')
plt.plot(gd_hist.history['val_accuracy'], label='validation accuracy')
plt.title("Training using Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend();plt.show()

plt.plot(sgd_hist.history['accuracy'], label='training accuracy')
plt.plot(sgd_hist.history['val_accuracy'], label='validation accuracy')
plt.title("Training using Stochastic Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend();plt.show()
