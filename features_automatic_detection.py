from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
import cv2


# a partir d'un parametre de blur et d'un parametre de selection, renvoie a partir d'une image en entreee le tableau des features et du skeletonization des features


# parametre sur lesquels jouer
ecart_type_blur = 5
parametre = 2.5


# parametres fixés
largeur_max = 256
largeur_min = 0
hauteur_max = 256
hauteur_min = 0
a = largeur_min
b = hauteur_min
theta = np.pi / 2
theta_1 = 0.0
theta_2 = 3 * np.pi / 4
theta_3 = np.pi / 4
nombre = largeur_max - largeur_min - 2


def selection_partie_image(image, largeur_min, largeur_max, hauteur_min, hauteur_max):
    return image[hauteur_min:hauteur_max, largeur_min:largeur_max]


def droite(a, b, theta):
    droite_0 = []
    if theta == np.pi / 4:
        for x in range(largeur_min, largeur_max):
            f = b - a + x
            if f >= hauteur_min and f < hauteur_max:
                droite_0.append([f, x])
        return droite_0
    elif theta == 3 * np.pi / 4:
        for x in range(largeur_min, largeur_max):
            f = b + a - x
            if f >= hauteur_min and f < hauteur_max:
                droite_0.append([f, x])
        return droite_0
    elif (theta < np.pi / 4 or theta > 7 * np.pi / 4) or (
        theta < 5 * np.pi / 4 and theta > 3 * np.pi / 4
    ):
        for x in range(largeur_min, largeur_max):
            f = int(np.tan(theta) * (x - a) + b)
            if f >= hauteur_min and f < hauteur_max:
                droite_0.append([f, x])
        return droite_0
    else:
        for y in range(hauteur_min, hauteur_max):
            x = int(1 / np.tan(theta) * (y - b) + a)
            if x >= largeur_min and x < largeur_max:
                droite_0.append([y, x])
        return droite_0


def rajoute_droite(a, b, theta, image):
    droit = droite(a, b, theta)
    for p in droit:
        image[p[0] - hauteur_min, p[1] - largeur_min] = (
            image[p[0] - hauteur_min, p[1] - largeur_min] - 50
        )
    return image


def rajoute_liste(lst, image):
    for p in lst:
        image[p[0], p[1]] = 250
    return ()


def valeur_image_sur_la_droite(a, b, theta, image):
    droit = droite(a, b, theta)
    lst = []
    for p in droit:
        lst.append(image[p[0], p[1]])
    return lst


def derivee(lst):
    n = len(lst)
    if n == 0 or n == 1:
        return []
    else:
        x = float(lst[1])
        y = float(lst[0])
        derivee = [x - y]
        for p in range(1, n - 1):
            x = float(lst[p + 1])
            y = float(lst[p - 1])
            s = (x - y) / 2
            derivee.append(s)
        x = float(lst[n - 1])
        y = float(lst[n - 2])
        derivee.append(x - y)
        return derivee


def affichage(a, b, theta, image, lst):
    for k in lst:
        courbe = valeur_image_sur_la_droite(a + k, b, theta, image)
        plt.plot([p for p in range(len(courbe))], courbe)
    plt.show()


def valeur_depassant_parametre(a, b, parametre, theta, image):
    droit = droite(a, b, theta)
    courbe = valeur_image_sur_la_droite(a, b, theta, image)
    test = derivee(derivee(courbe))
    accumulateur = []
    n = len(droit)
    if n == 0 or n == 1:
        return []
    else:
        for i in range(n):
            if test[i] >= parametre:
                accumulateur.append(droit[i])
        return accumulateur


def procedure_canny(image):
    image_blur = cv2.GaussianBlur(image, (ecart_type_blur, ecart_type_blur), 0)
    image_1 = image_blur.copy()
    image_2 = image_blur.copy()
    image_3 = image_blur.copy()
    image_4 = image_blur.copy()
    result = np.zeros((hauteur_max - hauteur_min, largeur_max - largeur_min), dtype=int)
    for r in range(nombre):
        lst = valeur_depassant_parametre(a + r, b, parametre, theta, image_1)
        rajoute_liste(lst, result)
        lst = valeur_depassant_parametre(a, b + r, parametre, theta_1, image_2)
        rajoute_liste(lst, result)
        lst = valeur_depassant_parametre(a + r, b + r, parametre, theta_2, image_3)
        rajoute_liste(lst, result)
        lst = valeur_depassant_parametre(a + r, b + r + 1, parametre, theta_2, image_3)
        rajoute_liste(lst, result)
        lst = valeur_depassant_parametre(
            largeur_max - largeur_min - 2 - r, b + 1 + r, parametre, theta_3, image_4
        )
        rajoute_liste(lst, result)
        lst = valeur_depassant_parametre(
            largeur_max - largeur_min - 2 - r, b + r, parametre, theta_3, image_4
        )
        rajoute_liste(lst, result)
    return result


def booleanisation(result):
    boolean = np.zeros(
        (hauteur_max - hauteur_min, largeur_max - largeur_min), dtype=bool
    )
    for i in range(hauteur_max - hauteur_min):
        for j in range(largeur_max - largeur_min):
            if result[i, j] == 250:
                boolean[i, j] = True
    return boolean


def points_importants(image):
    res = procedure_canny(image)
    resultat = booleanisation(res)
    return resultat


def tableau_features(image_de_base):
    okkk = skeletonize(points_importants(image_de_base))
    temp = np.zeros((largeur_max, hauteur_max), dtype=int)
    for p in range(largeur_max):
        for q in range(hauteur_max):
            if okkk[p, q]:
                temp[p, q] = 1
    return temp
