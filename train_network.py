# USAGE
# python train_network.py --dataset images --model perro_not_perro.model

#
# configura el extremo posterior de matplotlib para que las figuras se puedan guardar en el fondo
import matplotlib
matplotlib.use("Agg")


# importar los paquetes necesarios
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construir el argumento analizar y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# inicialice el número de épocas para entrenar, inicie la velocidad de aprendizaje y el tamaño del lote
EPOCHS = 25
INIT_LR = 1e-3
BS = 32


# Inicializar los datos y etiquetas.
print("[INFO] loading images...")
data = []
labels = []


# Agarra los caminos de imagen y los baraja aleatoriamente
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)


# bucle sobre las imágenes de entrada
for imagePath in imagePaths:
	# cargar la imagen, preprocesarla y almacenarla en la lista de datos
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	# extraiga la etiqueta de clase de la ruta de la imagen y actualice la lista de etiquetas
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "perro" else 0
	labels.append(label)


# escalar las intensidades de píxeles sin procesar al rango [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# divida los datos en divisiones de entrenamiento y prueba utilizando el 75% de los datos
# para entrenamiento y el 25% restante para prueba
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)


# Convertir las etiquetas de enteros a vectores.
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Construir el generador de imágenes para el aumento de datos.
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# inicializar el modelo
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# entrenar la red
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# guardar el modelo en el disco
print("[INFO] serializing network...")
model.save(args["model"])

# trazar la pérdida de entrenamiento y la precisión
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Pérdida de entrenamiento y precisión en Perro /No es Perro")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])