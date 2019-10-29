from __future__ import division
import numpy as np
import features_automatic_detection
import random


features_automatic_detection.ecart_type_blur = 3
features_automatic_detection.parametre = 3.3
couleur_de_affichage = 250


def coordinate_skel(image_skeletonize):
    n = len(image_skeletonize)
    m = len(image_skeletonize[0])
    lst = []
    for i in range(n):
        for j in range(m):
            if image_skeletonize[i, j] == 1:
                lst.append([i, j])
    return lst


def random_point(lst):
    n = len(lst)
    i = random.randint(0, n - 1)
    return [lst[i][1], lst[i][0]]


def affichage_features_on_image(image_de_bas, skel):
    n = len(image_de_bas)
    m = len(image_de_bas[0])
    somme = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            if skel[i, j] == 1:
                somme[i][j] = couleur_de_affichage
            else:
                somme[i][j] = image_de_bas[i, j]
    return somme


def affichage_points_random(image_de_bas, lst):
    im = image_de_bas.copy()
    for p in lst:
        im[p[0], p[1]] = couleur_de_affichage
    return im


def liste_de_voisins(point, liste_de_points):
    lst_voisin = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i != 0 or j != 0:
                if [point[0] + i, point[1] + j] in liste_de_points:
                    lst_voisin.append([point[0] + i, point[1] + j])
    return lst_voisin


# la condition i et j non nuls prend du temps de calcul


def trouve_une_arete(point, lst, accumulator):
    liste_voisin = liste_de_voisins(point, lst)
    n = len(liste_voisin)
    if n == 0:
        accumulator.append(point)
        lst.remove(point)
    elif n == 1:
        accumulator.append(point)
        voisin = liste_voisin[0]
        lst.remove(point)
        trouve_une_arete(voisin, lst, accumulator)
    return ()


def liste_des_aretes(lst):
    lst_1 = lst.copy()
    accumulateur = []
    i = 0
    while lst_1 != [] and i < 100000:
        part = []
        y = random_point(lst_1)
        trouve_une_arete([y[1], y[0]], lst_1, part)
        if part != []:
            accumulateur.append(part)
        i = i + 1
    while lst_1 != []:
        part = []
        y = random_point(lst_1)
        liste_voisin = liste_de_voisins([y[1], y[0]], lst_1)
        part.append([y[1], y[0]])
        lst_1.remove([y[1], y[0]])
        trouve_une_arete(liste_voisin[0], lst_1, part)
        if part != []:
            accumulateur.append(part)
    return accumulateur


def trouve_un_noeud(point, lst, accumulator):
    liste_voisin = liste_de_voisins(point, lst)
    accumulator.append(point)
    lst.remove(point)
    while liste_voisin != []:
        voisin = liste_voisin[0]
        trouve_un_noeud(voisin, lst, accumulator)
        liste_voisin = liste_de_voisins(point, lst)
    return ()


def liste_des_noeuds(lst):
    lst_1 = lst.copy()
    accumulateur = []
    while lst_1 != []:
        part = []
        y = random_point(lst_1)
        trouve_un_noeud([y[1], y[0]], lst_1, part)
        accumulateur.append(part)
    return accumulateur


# trouve une arete doit supprmier les elements de lst_1 et les rajouter dans une autre liste


def retirer_les_noeuds(lst):
    lst_noeuds = []
    for point in lst:
        lst_voisin = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i != 0 or j != 0:
                    if [point[0] + i, point[1] + j] in lst:
                        lst_voisin.append([point[0] + i, point[1] + j])
        n = len(lst_voisin)
        if n >= 3:
            lst_noeuds.append(point)
    for p in lst_noeuds:
        lst.remove(p)
    return lst_noeuds


def affichage_general(image_de_base, lst_arete):
    new_image = image_de_base.copy()
    for i in lst_arete:
        new_image = affichage_points_random(new_image, i)
    return new_image
