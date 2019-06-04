#instalamos todos estos paquetes en C:\Python\Python37\Scripts>

#pip3 install requests
#pip3 install imutils
#pip3 install opencv-python
#pip3 install matplotlib
#pip3 install keras
#pip3 install tensorflow
#pip3 install sklearn


# USAGE
# python download_images.py --urls paisajes.txt --output images/no-perro

# importar los paquetes necesarios
from imutils import paths
import argparse
import requests
import cv2
import os


# construir el argumento analizar y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
	help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())


# tome la lista de URL del archivo de entrada,
# luego inicialice el número total de imágenes descargadas hasta el momento
rows = open(args["urls"]).read().strip().split("\n")
total = 0

# busca los URLs
for url in rows:
	try:

		# intenta descargar la imagen
		r = requests.get(url, timeout=60)


		# guarda la imagen en el disco
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()

		# actualizar el contador
		print("[INFO] downloaded: {}".format(p))
		total += 1


	# manejar si se producen excepciones durante el proceso de descarga
	except:
		print("[INFO] error downloading {}...skipping".format(p))


# recorre las rutas de imagen que acabamos de descargar
for imagePath in paths.list_images(args["output"]):
	# Inicializa si la imagen debe ser borrada o no.
	delete = False


	# intenta cargar la imagen
	try:
		image = cv2.imread(imagePath)


		# Si la imagen es `Ninguna ', no podríamos cargarla correctamente.
        # del disco, así que bórralo

		if image is None:
			print("None")
			delete = True

	# Si OpenCV no puede cargar la imagen, entonces la imagen es probable.
	# corrupto por lo que deberíamos eliminarlo
	except:
		print("Except")
		delete = True

	# Compruebe si la imagen debe ser borrada
	if delete:
		print("[INFO] deleting {}".format(imagePath))
		os.remove(imagePath)