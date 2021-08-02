# Importieren der notwendigen Bibliotheken
import os
import logging
import tensorflow as tf
import numpy as np
import time

# Tensorflow logging minifieren
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "6"  # FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Dauer des Durchlaufs messen
starttime = time.time()

# https://de.wikipedia.org/wiki/MNIST-Datenbank
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

print("Duration: ", time.time() - starttime)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])

# Erstellen des gespeicherten Modells
tf.keras.models.save_model(model, os.path.abspath("saved_model"))

if "nt" in os.name:
    # Modell convertieren
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Modell speichern
    with open(os.path.abspath('model.tflite'), 'wb') as f:
        f.write(tflite_model)
