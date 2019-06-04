# USAGE
# python test_network.py --model perro_not_perro.model --image images/ejemplos/1.jpg

# importar los paquetes necesarios
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


# construir el argumento analizar y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# carga la imagen
image = cv2.imread(args["image"])
orig = image.copy()

# preprocesar la imagen para su clasificación
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# cargar la red neuronal convolucional entrenada
print("[INFO] loading network...")
model = load_model(args["model"])


# clasificar la imagen de entrada
(noperro, perro) = model.predict(image)[0]


# construir la etiqueta
label = " Es perro" if perro > noperro else "No es perro"
proba = perro if perro > noperro else noperro
label = "{}: {:.2f}%".format(label, proba * 100)


# dibujar la etiqueta en la imagen
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)


# muestra la imagen de salida
cv2.imshow("Output", output)
cv2.waitKey(0)