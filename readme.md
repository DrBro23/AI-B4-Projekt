# Künstliche Intelligenz in Kombination mit einem Raspberry Pi

_von Robin Prillwitz, Simon Obermeier und Sven Menzel_

## Installationen

Voraussetzung ist Python mit Version 3.

Es wird zu dem gewünschten Speicherordner navigiert.
Eine virtuelle Umgebung erstellen und gestartet:

```bash
git clone https://github.com/DrBro23/AI-B4-Projekt.git
python -m venv venv
venv\Scripts\activate
```

Die Aktivierung des `venv` varriert je nach shell.
- windows batch oder powershell: `venv\Scripts\activate`
- bash: `source venv\bin\activate`

pip aktualisieren (optional):
```bash
pip install --upgrade pip
```

requirements installieren:
```bash
pip install -r requirements.txt
```


## Training des Modells

- [Mit Mnist Datensatz](./mnist.md)


## Quellen

- https://www.tensorflow.org/install/pip
- https://pypi.org/project/tflite-model-maker/
