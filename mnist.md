# Training mit dem Mnist Datensatz

[Source Code](./src/train_mnist.py)

Ausführen mit:
```bash
python src/train_mnist.py
```

## Was ist Mnist?

Die MNIST-Datenbank _(Modified National Institute of Standards and Technology database)_ ist eine öffentlich verfügbare Datenbank von handgeschriebenen Ziffern ([Wikipedia](https://de.wikipedia.org/wiki/MNIST-Datenbank)).

## Was macht der Code?

Es trainiert ein Neuronales Netzwerk (NN) zur Erkennung von den Mnist Bildern.
Jede handgeschriebene Ziffer wird durch das Neuronale Netz erkannt und ausgegeben.

Die Bilder werden zuerst "normalisiert" (hier: in ein einheitliches Format gebracht)

### Training

Anhand von einer Vielzahl von Bildern wird das neuronale Netz darauf trainiert, sich Formen und Konturen so zu merken, damit es Zahlen von 0 bis 9 erkennen kann.

### Speichern

TODO

#### `savedmodel`

TODO

#### `tflite`

TODO

## Ergebnis

| Gerät                 | Zeit      | CPU                       | Kerne (Threads)   | CPU Takt  | RAM       |
| ---                   | ---       | ---                       | ---               | ---       | ---       |
| Apple MacBook Pro     | 3.9s      | M1                        | ?                 | ?         | 16GB      |
| Desktop Pc 1          | 7.8s      | Ryzen 5 3600XT            | 6 (12)            | 4.50 GHz  | 16GB      |
| Laptop Lenovo         | 12.6s     | i5 8250U                  | 4 (8)             | 3.40 GHz  | 8GB       |
| Acer Aspire e15       | 49.3s     | AMD A10-7300 Radeon R6    | 4                 | 1.9 GHz   | 8GB       |
| Raspberry Pi 4        | 74.9s     | BCM2711 Cortex A72        | 4                 | 1.5 GHz   | 8GB       |
---
