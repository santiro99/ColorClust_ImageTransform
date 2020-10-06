import cv2
import sys
import numpy as np
import os

class geomTransform:
    def __init__(self, pts1, pts2):
        self.pts1 = pts1
        self.pts2 = pts2
        self.M_affine = cv2.getAffineTransform(pts1, pts2)

    def affineTransform(self, image):
        # Computar transformación afin de la imagen ingresada como parámetro y mostrar
        image_affine = cv2.warpAffine(image, self.M_affine, image.shape[:2])
        cv2.imshow("Image Affine", image_affine)
        cv2.waitKey(0)

    def estSimilarity(self):
        # Estimar parámetros de escala, rotación y traslación a partir de la matriz de transformación afin
        s0 = np.sqrt(self.M_affine[0, 0] ** 2 + self.M_affine[1, 0] ** 2)
        s1 = np.sqrt(self.M_affine[0, 1] ** 2 + self.M_affine[1, 1] ** 2)
        theta = -np.arctan(self.M_affine[1, 0] / self.M_affine[0, 0])
        x0 = (self.M_affine[0, 2] * np.cos(theta) - self.M_affine[1, 2] * np.sin(theta)) / s0
        x1 = (self.M_affine[0, 2] * np.sin(theta) + self.M_affine[1, 2] * np.cos(theta)) / s1

        self.M_sim = np.float32(
            [[s0 * np.cos(theta), s1 * np.sin(theta), (s0 * x0 * np.cos(theta) + s1 * x1 * np.sin(theta))],
             [-s0 * np.sin(theta), s1 * np.cos(theta), (s1 * x1 * np.cos(theta) - s0 * x0 * np.sin(theta))]])

    def similarityTransform(self, image):
        # Computar transformación de similitud de la imagen ingresada como parámetro y mostrar
        image_similarity = cv2.warpAffine(image, self.M_sim, image.shape[:2])
        cv2.imshow("Image Similarity", image_similarity)
        cv2.waitKey(0)

    def similarityError(self):
        # Calcular transformación de similitud sobre los puntos1 guardados en el objeto y calcular
        # error respecto a puntos2
        M_pts = np.append(self.M_sim, np.array([[0, 0, 1]]), axis=0)
        pts = np.append(self.pts1.transpose(), np.array([[1, 1, 1]]), axis=0)
        pts_transform = np.matmul(M_pts, pts)
        pts_transform = pts_transform[:-1,:].transpose()

        error = np.linalg.norm(pts_transform - self.pts2, axis=1)

        # print(pts_transform)
        # print(self.pts2)
        print('Error = ', error)


def capturePoints(event, x, y, flags, params):
    # Función para capturar eventos con el mouse
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        # print(refPt)

if __name__ == '__main__':
    # Path de imagen1 e imagen2
    path = sys.argv[1]
    image_name1 = sys.argv[2]
    image_name2 = sys.argv[3]
    path_file1 = os.path.join(path, image_name1)
    path_file2 = os.path.join(path, image_name2)

    # Leer imágenes y garantizar que sean cuadradas
    image1 = cv2.imread(path_file1)
    image1 = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_CUBIC)
    image2 = cv2.imread(path_file2)
    image2 = cv2.resize(image2, (512, 512), interpolation=cv2.INTER_CUBIC)

    # Mostrar imagen 1 y capturar puntos indicados con el mouse
    refPt = []
    cv2.imshow('Image1', image1)
    cv2.setMouseCallback('Image1', capturePoints)
    cv2.waitKey(0)
    puntos1 = refPt
    pts1 = np.float32(puntos1)

    # Mostrar imagen 2 y capturar puntos indicados con el mouse
    refPt = []
    cv2.imshow('Image2', image2)
    cv2.setMouseCallback('Image2', capturePoints)
    cv2.waitKey(0)
    puntos2 = refPt
    pts2 = np.float32(puntos2)

    # Calcular matriz de transformación afin (al crear objeto), transformar imagen 1 y mostrar
    affine = geomTransform(pts1, pts2)
    affine.affineTransform(image1)

    # Estimar parámetros y calcular matriz de similitud
    affine.estSimilarity()
    # Mostrar imagen1 transformada por matriz de similitud estimada
    affine.similarityTransform(image1)

    # Calcular transformación de similitud sobre los puntos capturados de la imagen1 y mostrar error
    # respecto a los puntos capturados de la imagen 2
    affine.similarityError()


