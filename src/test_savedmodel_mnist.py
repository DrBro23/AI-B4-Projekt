import os
import random
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

index = 0
doShow = True
delayTime = 3

durationAcc = 0
durationAccCount = 0

# Lade DAten vom Mnist Datensatz
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Laden des TFLite Models und Zuordnung der tensors.
model = tf.keras.models.load_model(os.path.abspath("./saved_model"))
model.compile()
# infer = model.signatures['serving_default']

while True:
    # Eingabe und erwartete Ausgabedaten setzen
    input_data = np.array(x_train[index], dtype=np.float32)
    actual = y_train[index]

    starttime = time.time()

    # Inferenz
    output_data =  model.predict(np.resize(input_data, (1, 28, 28)))

    duration = time.time() - starttime
    durationAcc += duration
    durationAccCount += 1

    # Interpretiere Ausgabe
    num = output_data[0].argmax(axis=0)

    print(output_data)
    print(f"\n{index}: NN: {num} Actual: {actual} Duration: {duration} Avg duration: {durationAcc / durationAccCount}\n")

    # Zeige Eingabebild für den Benutzer
    if doShow:
        plt.imshow(input_data)
        plt.show(block=False)
        plt.pause(delayTime)
        plt.close()
    else:
        time.sleep(delayTime)

    index += 1
