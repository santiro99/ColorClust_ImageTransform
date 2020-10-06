import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

class colorSeg:
    def __init__(self, path, method):
        # Recibe path de la imagen la lee y las guarda en el objeto junto
        # con el método de clustering ingresado como parámetro
        self.image = cv2.imread(path)
        self.method = method

        # Cambia imágen a representación RGB y a flotante
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = np.array(self.image, dtype=np.float64) / 255

        # Guarda en el objeto tamaño de la imagen y verifica que tenga 3 canales
        self.rows, self.cols, ch = self.image.shape
        assert ch == 3
        # Guarda la imagen convertida a un arreglo de 2D
        self.image_array = np.reshape(self.image, (self.rows * self.cols, ch))

    def clustering(self, n_color):
        # Computar segmentación de color para valores n_color según el método guardado en el objeto
        self.n_color = n_color
        image_array_sample = shuffle(self.image_array, random_state=0)[:10000]
        if self.method == 'gmm':
            model = GMM(n_components=n_color).fit(image_array_sample)
            self.labels = model.predict(self.image_array)
            self.centers = model.means_
        else:
            model = KMeans(n_clusters=n_color, random_state=0).fit(image_array_sample)
            self.labels = model.predict(self.image_array)
            self.centers = model.cluster_centers_

    def calc_dist(self):
        # Calcular suma de distancias intracluster para un valor de n_color
        intracluster = 0
        for label in range(self.centers.shape[0]):
            vector_label = self.image_array[self.labels == label]
            resta = vector_label - self.centers[label]
            magnitude = np.linalg.norm(resta, axis=1)
            distancia = np.sum(magnitude)
            intracluster = intracluster + distancia
        return intracluster


    def recreate_image(self, fig):
        # Mostrar imagen segmentada para un valor de n_color
        d = self.centers.shape[1]
        image_clusters = np.zeros((self.rows, self.cols, d))
        label_idx = 0
        for i in range(self.rows):
            for j in range(self.cols):
                image_clusters[i][j] = self.centers[self.labels[label_idx]]
                label_idx += 1

        plt.figure(fig)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(self.n_color, self.method))
        plt.imshow(image_clusters)
        plt.show()


if __name__ == '__main__':
    # Path de la imagen
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    # Método de clustering kmeans o gmm
    method = 'gmm'

    imagen = colorSeg(path_file, method)    # Crear objeto con parámetros indicados

    # Computar la distancia intracluster para cada número de clusters de 1 a 10
    num_clusters = 10
    distancias = np.zeros((num_clusters, 1))
    for n_color in range(1, num_clusters + 1):
        imagen.clustering(n_color)     # Para n_color calcular clustering
        imagen.recreate_image(n_color) # Mostrar imagen segmentada por n_color
        distancias[n_color - 1] = imagen.calc_dist() # Calcular distancias intracluster para n_color clusters

    # Graficar suma de distancias intra-cluster vs n_color
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, num_clusters+1), distancias, marker='o')
    plt.title('Suma de distancias intracluster vs Numero colores, GMM')
    plt.xlabel('Numero de colores')
    plt.ylabel('SUma de distancias intracluster')
    plt.xticks(np.arange(0, num_clusters+1, 1))
    plt.show()


