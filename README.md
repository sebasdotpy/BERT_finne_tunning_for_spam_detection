# BERT fine tunning for text classification
En este proyecto se hace uso de la librería `transformers` para descargar y reentrenar el modelo [BERT](https://arxiv.org/abs/1810.04805) en búsqueda de solucionar un problema de clasificación y detectar si un mensaje es o no SPAM/PHISHING
### INSTALACION

---

Descargamos el repositorio

```sh
$ git clone https://github.com/sebasdotpy/BERT_finne_tunning_for_spam_detection
$ cd BERT_finne_tunning_for_spam_detection
```


Para tener una correcta instalación de este proyecto usaremos Anaconda, primero creando el ambiente virtual requerido

```sh
# creamos el ambiente virtual para trabajar y entramos en el
$ conda create -n ENV_NAME python=3.8 -c conda-forge
$ conda activate ENV_NAME

# instalamos pytorch
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# instalacion del resto de librerias
$ pip install -r requirements.txt
```


Nos posicionamos en la carpeta llamada `app` para poder iniciar el servidor y ejecutamos `main.py`

```sh
$ cd app
$ python main.py
```

Con todo lo anterior hecho tendremos ejecutándose nuestra API para la clasificación de texto usando el modelo BERT
