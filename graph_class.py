import pickle
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import random
import edges_and_nodes_creation


indice_de_confiance_1 = 10
indice_de_confiance_2 = 3
indice_de_confiance_3 = 0.1
indice_de_confiance_4 = 4
indice_de_confiance_5 = 0.2
indice_de_confiance_7 = 10
nombre_iteration = 300000
parametre_trois_aretes_un_point = 0.1
longueur_de_la_courbe = 15
coeff_ouverture_courbe = 2

rapport_libre_sur_courbe_fusion = 2

param_courbure = 0.1
param_angle = 1.2

edges_and_nodes_creation.couleur_de_affichage = -40

liste_des_mauvais_choix = []
liste_des_super_cercles = []


class Super_arete:
    def __init__(self):
        self.liste_des_points = []
        self.extremités = [[0.0, 0.0], [0.0, 0.0]]
        self.tangente = [0.0, 0.0]
        self.longueur = 0
        self.angle = []
        self.angle_moindre_carrés = []
        self.courbure = [0.0, 0.0]
        self.correlation_courbure = 1.0

    def print_Super_arete(self):
        print("Arete", self.liste_des_points)
        print("Extremités", self.extremités)
        print("Tangente", self.tangente)
        print("Longueur", self.longueur)
        print("Angle", self.angle)
        print("Angle_regression", self.angle_moindre_carrés)
        print("Courbure", self.courbure)
        print("Correlation_courbure", self.correlation_courbure)
        print()

    def implementer_extremite(self):
        liste = self.liste_des_points
        liste_extremites = []
        for p in liste:
            voisins = liste_de_voisins(p, liste)
            if len(voisins) == 1:
                liste_extremites.append(p)
        if len(liste_extremites) == 0:
            liste_extremites = [liste[0], liste[-1]]
        self.extremités = liste_extremites


class Super_noeud:
    def __init__(self):
        self.liste_des_points = []
        self.liste_des_super_aretes_connectées = []
        self.extremités = []
        self.nombre_aretes = 0

    def print_Super_noeud(self):
        print("Elements_du_noeud", self.liste_des_points)
        print()
        print_liste_Super_aretes(self.liste_des_super_aretes_connectées)
        print("Extremités_noeud", self.extremités)
        print("Nombre daretes", self.nombre_aretes)
        print()


class Cercle:
    def __init__(self):
        self.liste_des_points = []
        self.rayon = 0.0
        self.centre = [0.0, 0.0]


def afficher_des_elements(liste_des_aretes):
    lst = []
    for p in liste_des_aretes:
        lst = lst + p.liste_des_points
    affichage_features_on_image(image, lst)
    print()
    return ()


def angle(point_1, point_2):
    return np.angle((point_2[1] - point_1[1]) + (point_1[0] - point_2[0]) * 1j)


def super_noeuds_egaux(super_noeuds_1, super_noeuds_2):
    if super_noeuds_1.liste_des_points == super_noeuds_2.liste_des_points:
        return True
    else:
        return False


def super_aretes_egaux(super_aretes_1, super_aretes_2):
    if super_aretes_1.liste_des_points == super_aretes_2.liste_des_points:
        return True
    else:
        return False


def retirer_un_element_liste_de_super_noeuds(noeuds, liste_de_noeuds):
    for p in liste_de_noeuds:
        if super_noeuds_egaux(p, noeuds):
            liste_de_noeuds.remove(p)
    return ()


def retirer_un_element_liste_de_super_aretes(aretes, liste_de_aretes):
    for p in liste_de_aretes:
        if super_aretes_egaux(p, aretes):
            liste_de_aretes.remove(p)
    return ()


def ajouter_un_element_liste_de_super_aretes(aretes, liste_de_aretes):
    liste_de_aretes = liste_de_aretes.append(aretes)
    return ()


def ajouter_un_element_liste_de_super_noeuds(noeud, liste_des_noeuds):
    liste_des_noeuds = liste_des_noeuds.append(noeud)
    return ()


def print_liste_Super_aretes(liste_de_super_aretes):
    for p in liste_de_super_aretes:
        p.print_Super_arete()


def print_liste_Super_noeuds(liste_de_super_noeuds):
    for p in liste_de_super_noeuds:
        p.print_Super_noeud()


def implementer_une_liste_de_super_aretes(liste_arete):
    liste_de_super_aretes = []
    for p in liste_arete:
        new_super_arete = Super_arete()
        new_super_arete.liste_des_points = p
        new_super_arete.longueur = len(p)
        liste_de_super_aretes.append(new_super_arete)
    return liste_de_super_aretes


def implementer_extremites_liste_super_aretes(liste_super_aretes):
    for p in liste_super_aretes:
        p.implementer_extremite()


def implementer_une_liste_de_super_noeuds(liste_noeud):
    liste_de_super_noeuds = []
    for p in liste_noeud:
        new_super_noeud = Super_noeud()
        new_super_noeud.liste_des_points = p
        liste_de_super_noeuds.append(new_super_noeud)
    return liste_de_super_noeuds


def liste_de_voisins(point, liste_de_points):
    lst_voisin = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i != 0 or j != 0:
                if [point[0] + i, point[1] + j] in liste_de_points:
                    lst_voisin.append([point[0] + i, point[1] + j])
    return lst_voisin


def enlever_doublon(liste):
    n = len(liste)
    new = []
    if n == 1:
        new.append(liste[0])
        return new
    else:
        for i in range(0, n - 1):
            if liste[i] != liste[i + 1]:
                new.append(liste[i])
        new.append(liste[n - 1])
    return new


def liste_de_voisins_procedure(point, accumulateur):
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            accumulateur.append([point[0] + i, point[1] + j])
    return accumulateur


def implementer_aretes_des_noeuds(liste_de_super_noeud, liste_de_super_arete):
    for p in liste_de_super_noeud:
        extremites_total = []
        aretes_voisines_des_noeuds = []
        pixel = p.liste_des_points
        acc = []
        s = 0
        for point in pixel:
            liste_de_voisins_procedure(point, acc)
        for k in liste_de_super_arete:
            extremite = []
            extremite_1 = k.extremités[0]
            extremite_2 = k.extremités[1]
            if (extremite_1 in acc) or (extremite_2 in acc):
                aretes_voisines_des_noeuds.append(k)
                extremite = [extremite_1 in acc, extremite_2 in acc]
                s = s + 1
            if len(extremite) != 0:
                extremites_total.append(extremite)
        p.liste_des_super_aretes_connectées = aretes_voisines_des_noeuds
        p.extremités = extremites_total
        p.nombre_aretes = s
    return ()


def trouver_angle(point_1, point_2):
    if point_1[1] == point_2[1]:
        if point_1[0] <= point_2[0]:
            return -np.pi / 2
        else:
            return np.pi / 2
    elif point_1[0] == point_2[0]:
        if point_1[1] <= point_2[1]:
            return 0.0
        else:
            return np.pi
    elif point_2[1] > point_1[1]:
        return -np.arctan((point_2[0] - point_1[0]) / (point_2[1] - point_1[1]))
    else:
        return -np.arctan((point_2[0] - point_1[0]) / (point_2[1] - point_1[1])) + np.pi


def implementer_angle_super_aretes(super_aretes):
    liste_de_point = super_aretes.liste_des_points
    n = len(liste_de_point)
    liste_des_angles = np.zeros(n, dtype=float)
    if n == 1:
        liste_des_angles[0] = 20.0
    else:
        liste_des_angles[0] = trouver_angle(liste_de_point[0], liste_de_point[1])
        for i in range(1, n - 1):
            liste_des_angles[i] = trouver_angle(
                liste_de_point[i - 1], liste_de_point[i + 1]
            )
            liste_des_angles[i] = correction_modulo_2pi(
                liste_des_angles[i - 1], liste_des_angles[i]
            )
        liste_des_angles[n - 1] = trouver_angle(
            liste_de_point[n - 2], liste_de_point[n - 1]
        )
        liste_des_angles[n - 1] = correction_modulo_2pi(
            liste_des_angles[n - 2], liste_des_angles[n - 1]
        )
    return liste_des_angles


def correction_modulo_2pi(a, b):
    dist_0 = abs(b - 4 * 2 * np.pi - a)
    k = -4
    for i in range(-4, 5):
        dist_1 = abs(b + i * 2 * np.pi - a)
        if dist_1 <= dist_0:
            dist_0 = dist_1
            k = i
    return b + k * 2 * np.pi


def correction_modulo_pi(a, b):
    dist_0 = abs(b - 4 * np.pi - a)
    k = -4
    for i in range(-4, 5):
        dist_1 = abs(b + i * np.pi - a)
        if dist_1 <= dist_0:
            dist_0 = dist_1
            k = i
    return b - k * np.pi


def implementer_angle_liste_super_aretes(liste_super_aretes):
    for p in liste_super_aretes:
        p.angle = implementer_angle_super_aretes(p)
    return ()


def regression_lineaire_des_angles(super_arete):
    liste = super_arete.angle
    n = len(liste)
    x = range(0, n)
    if n == 1:
        return [liste, 0.0, 1.0]
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, liste)
        regression = [slope * x[i] + intercept for i in range(0, n)]
        return [regression, slope, abs(r_value)]


def implementer_regression_liste_super_aretes(liste_super_aretes):
    for p in liste_super_aretes:
        regress = regression_lineaire_des_angles(p)
        p.angle_moindre_carrés = regress[0]
        p.courbure = [regress[1], regress[1]]
        p.correlation_courbure = regress[2]
    return ()


def implementer_tangente_liste_super_aretes(liste_super_aretes):
    for p in liste_super_aretes:
        regression = p.angle_moindre_carrés
        n = len(regression)
        p.tangente[0] = regression[0] + np.pi
        courbure_en_absolue = p.courbure[0]
        p.courbure[0] = -courbure_en_absolue
        p.tangente[1] = regression[n - 1]
        p.courbure[1] = courbure_en_absolue
    return ()


def affichage_features_on_image(image_de_bas, liste_skel):
    n = len(image_de_bas)
    m = len(image_de_bas[0])
    somme = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            if [i, j] in liste_skel:
                somme[i][j] = edges_and_nodes_creation.couleur_de_affichage
            else:
                somme[i][j] = image_de_bas[i, j]
    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(8, 8),
        sharex=False,
        sharey=False,
        subplot_kw={"adjustable": "box-forced"},
    )
    ax1.imshow(somme)
    plt.show()
    return somme


def distance_modulo_2pi(a, b, c, d):
    val = (
        abs(a - correction_modulo_2pi(a, b - np.pi)) ** 2
        + abs(c - correction_modulo_2pi(c, d - np.pi)) ** 2
    )
    return val


def distance_modulo_2pi_deux_valeurs(a, b):
    val = abs(a - correction_modulo_2pi(a, b - np.pi)) ** 2
    return val


def couple_deux_a_deux_tangente(super_noeud, indicateur_confiance):
    [
        tangente,
        mauvaises_extremités,
        super_arete,
        courbure,
    ] = renvoie_les_tangentes_et_les_mauvaises_extremités(super_noeud)
    [couple, mauvaises_extremité] = test_si_une_arete_est_de_taille_1(
        tangente, mauvaises_extremités, super_arete
    )
    lst = []
    indice_de_confiance = 11
    if mauvaises_extremité == [0.0, 0.0, 0.0, 0.0]:
        valeur_0 = distance_modulo_2pi(
            tangente[0], tangente[1], tangente[2], tangente[3]
        )
        valeur_1 = distance_modulo_2pi(
            tangente[0], tangente[2], tangente[1], tangente[3]
        )
        valeur_2 = distance_modulo_2pi(
            tangente[0], tangente[3], tangente[1], tangente[2]
        )
        valeur = valeur_0
        couple = [[super_arete[0], super_arete[1]], [super_arete[2], super_arete[3]]]
        mauvaises_extremité = [
            mauvaises_extremités[0],
            mauvaises_extremités[1],
            mauvaises_extremités[2],
            mauvaises_extremités[3],
        ]
        if (
            distance_modulo_2pi(tangente[0], tangente[2], tangente[1], tangente[3])
            < valeur
        ):
            valeur = valeur_1
            couple = [
                [super_arete[0], super_arete[2]],
                [super_arete[1], super_arete[3]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[0],
                mauvaises_extremités[2],
                mauvaises_extremités[1],
                mauvaises_extremités[3],
            ]
        if (
            distance_modulo_2pi(tangente[0], tangente[3], tangente[1], tangente[2])
            < valeur
        ):
            valeur = valeur_2
            couple = [
                [super_arete[0], super_arete[3]],
                [super_arete[1], super_arete[2]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[0],
                mauvaises_extremités[3],
                mauvaises_extremités[1],
                mauvaises_extremités[2],
            ]
        if valeur != 0:
            lst.append(
                distance_modulo_2pi(tangente[0], tangente[1], tangente[2], tangente[3])
                / valeur
            )
            lst.append(
                distance_modulo_2pi(tangente[0], tangente[2], tangente[1], tangente[3])
                / valeur
            )
            lst.append(
                distance_modulo_2pi(tangente[0], tangente[3], tangente[1], tangente[2])
                / valeur
            )
        lst.remove(1)
        indice_de_confiance = min(lst)
        if indice_de_confiance > indicateur_confiance and (
            (valeur_1 > 1 and valeur_2 > 1)
            or (valeur_0 > 1 and valeur_2 > 1)
            or (valeur_1 > 1 and valeur_2 > 1)
        ):
            return [couple, mauvaises_extremité, indice_de_confiance]
        else:
            lst = []
            valeur = 100000
            if critere_de_courbure_gentil(
                courbure[0], courbure[1]
            ) and critere_de_courbure_gentil(courbure[2], courbure[3]):
                valeur = distance_modulo_2pi(
                    tangente[0], tangente[1], tangente[2], tangente[3]
                )
                couple = [
                    [super_arete[0], super_arete[1]],
                    [super_arete[2], super_arete[3]],
                ]
                mauvaises_extremité = [
                    mauvaises_extremités[0],
                    mauvaises_extremités[1],
                    mauvaises_extremités[2],
                    mauvaises_extremités[3],
                ]
            if (
                distance_modulo_2pi(tangente[0], tangente[2], tangente[1], tangente[3])
                < valeur
                and critere_de_courbure_gentil(courbure[0], courbure[2])
                and critere_de_courbure_gentil(courbure[1], courbure[3])
            ):
                valeur = distance_modulo_2pi(
                    tangente[0], tangente[2], tangente[1], tangente[3]
                )
                couple = [
                    [super_arete[0], super_arete[2]],
                    [super_arete[1], super_arete[3]],
                ]
                mauvaises_extremité = [
                    mauvaises_extremités[0],
                    mauvaises_extremités[2],
                    mauvaises_extremités[1],
                    mauvaises_extremités[3],
                ]
            if (
                distance_modulo_2pi(tangente[0], tangente[3], tangente[1], tangente[2])
                < valeur
                and critere_de_courbure_gentil(courbure[0], courbure[3])
                and critere_de_courbure_gentil(courbure[1], courbure[2])
            ):
                valeur = distance_modulo_2pi(
                    tangente[0], tangente[3], tangente[1], tangente[2]
                )
                couple = [
                    [super_arete[0], super_arete[3]],
                    [super_arete[1], super_arete[2]],
                ]
                mauvaises_extremité = [
                    mauvaises_extremités[0],
                    mauvaises_extremités[3],
                    mauvaises_extremités[1],
                    mauvaises_extremités[2],
                ]
            if valeur != 100000:
                if critere_de_courbure_gentil(
                    courbure[0], courbure[1]
                ) and critere_de_courbure_gentil(courbure[2], courbure[3]):
                    lst.append(
                        distance_modulo_2pi(
                            tangente[0], tangente[1], tangente[2], tangente[3]
                        )
                        / valeur
                    )
                if critere_de_courbure_gentil(
                    courbure[0], courbure[2]
                ) and critere_de_courbure_gentil(courbure[1], courbure[3]):
                    lst.append(
                        distance_modulo_2pi(
                            tangente[0], tangente[2], tangente[1], tangente[3]
                        )
                        / valeur
                    )
                if critere_de_courbure_gentil(
                    courbure[0], courbure[3]
                ) and critere_de_courbure_gentil(courbure[1], courbure[2]):
                    lst.append(
                        distance_modulo_2pi(
                            tangente[0], tangente[3], tangente[1], tangente[2]
                        )
                        / valeur
                    )
                if lst == []:
                    return [couple, mauvaises_extremité, indice_de_confiance]
                else:
                    if 1.0 in lst:
                        lst.remove(1.0)
                        if lst == []:
                            indice_de_confiance = 50
                        else:
                            indice_de_confiance = min(lst)

                        return [couple, mauvaises_extremité, indice_de_confiance]
            else:
                valeur = 0
                indice_de_confiance = 0
                lst = []
                return [couple, mauvaises_extremité, indice_de_confiance]
    else:
        return [couple, mauvaises_extremité, indice_de_confiance]


def couple_pour_3_aretes(liste_des_noeuds, super_noeud, indice_de_confiance):
    [
        tangente,
        mauvaises_extremités,
        super_arete,
        courbure,
    ] = renvoie_les_tangentes_et_les_mauvaises_extremités(super_noeud)
    couple = []
    mauvaises_extremité = []
    courbure_0 = []
    valeurs = []
    [
        couple,
        mauvaises_extremité,
        valeurs,
        courbure_0,
    ] = test_si_une_arete_est_de_taille_1_3_aretes(
        liste_des_noeuds, tangente, mauvaises_extremités, super_arete, courbure
    )
    if mauvaises_extremité == []:
        valeur_2 = distance_modulo_2pi_deux_valeurs(tangente[0], tangente[1])
        valeur_1 = distance_modulo_2pi_deux_valeurs(tangente[0], tangente[2])
        valeur_0 = distance_modulo_2pi_deux_valeurs(tangente[1], tangente[2])
        correlation_courbure_0 = fusion_deux_super_aretes(
            super_arete[1],
            super_arete[2],
            mauvaises_extremités[1],
            mauvaises_extremités[2],
        ).correlation_courbure
        correlation_courbure_1 = fusion_deux_super_aretes(
            super_arete[0],
            super_arete[2],
            mauvaises_extremités[0],
            mauvaises_extremités[2],
        ).correlation_courbure
        correlation_courbure_2 = fusion_deux_super_aretes(
            super_arete[0],
            super_arete[1],
            mauvaises_extremités[0],
            mauvaises_extremités[1],
        ).correlation_courbure

        if valeur_2 > 1.5 or correlation_courbure_2 < 0.5:
            if correlation_courbure_0 > param_courbure and valeur_0 < param_angle:
                valeurs.append(valeur_0)
                couple.append([super_arete[1], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[1], mauvaises_extremités[2]]
                )
                courbure_0.append([courbure[1], courbure[2]])
            if correlation_courbure_1 > param_courbure and valeur_1 < param_angle:
                valeurs.append(valeur_1)
                couple.append([super_arete[0], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[2]]
                )
                courbure_0.append([courbure[0], courbure[2]])
        elif valeur_1 > 1.5 or correlation_courbure_1 < 0.5:
            if correlation_courbure_0 > param_courbure and valeur_0 < param_angle:
                valeurs.append(valeur_0)
                couple.append([super_arete[1], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[1], mauvaises_extremités[2]]
                )
                courbure_0.append([courbure[1], courbure[2]])
            if correlation_courbure_2 > param_courbure and valeur_2 < param_angle:
                valeurs.append(valeur_2)
                couple.append([super_arete[0], super_arete[1]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[1]]
                )
                courbure_0.append([courbure[0], courbure[1]])
        elif valeur_0 > 1.5 or correlation_courbure_0 < 0.5:
            if correlation_courbure_2 > param_courbure and valeur_2 < param_angle:
                valeurs.append(valeur_2)
                couple.append([super_arete[0], super_arete[1]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[1]]
                )
                courbure_0.append([courbure[0], courbure[1]])
            if correlation_courbure_1 > param_courbure and valeur_1 < param_angle:
                valeurs.append(valeur_1)
                couple.append([super_arete[0], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[2]]
                )
                courbure_0.append([courbure[0], courbure[2]])
    return [couple, mauvaises_extremité, valeurs, courbure_0]


def couple_pour_2_aretes(liste_des_noeuds, super_noeud, indice_de_confiance):
    [
        tangente,
        mauvaises_extremités,
        super_arete,
        courbure,
    ] = renvoie_les_tangentes_et_les_mauvaises_extremités(super_noeud)
    couple = []
    mauvaises_extremité = []
    courbure_0 = []
    valeurs = []
    if mauvaises_extremité == []:
        valeur = distance_modulo_2pi_deux_valeurs(tangente[0], tangente[1])
        if valeur < indice_de_confiance and critere_de_courbure(
            courbure[0], courbure[1]
        ):
            valeurs.append(valeur)
            couple.append([super_arete[0], super_arete[1]])
            mauvaises_extremité.append(
                [mauvaises_extremités[0], mauvaises_extremités[1]]
            )
            courbure_0.append([courbure[0], courbure[1]])
    return [couple, mauvaises_extremité, valeurs, courbure_0]


def critere_de_courbure(courbure_1, courbure_2):
    if courbure_1 == 0.0 or courbure_2 == 0.0:
        return True
    elif courbure_1 * courbure_2 < 0:
        return True
    elif (courbure_1 / courbure_2) > indice_de_confiance_4 or (
        courbure_2 / courbure_1
    ) > indice_de_confiance_4:
        return True
    else:
        return False


def critere_de_courbure_gentil(courbure_1, courbure_2):
    if courbure_1 == 0.0 or courbure_2 == 0.0:
        return True
    elif courbure_1 * courbure_2 < 0:
        return True
    elif (courbure_1 / courbure_2) > indice_de_confiance_7 or (
        courbure_2 / courbure_1
    ) > indice_de_confiance_7:
        return True
    else:
        return False


def test_si_une_arete_est_de_taille_1(tangente, mauvaises_extremités, super_arete):
    [couple, mauvaises_extremités] = [
        [[Super_arete(), Super_arete()], [Super_arete(), Super_arete()]],
        [0.0, 0.0, 0.0, 0.0],
    ]
    if len(super_arete[0].liste_des_points) == 1:
        valeur = distance_modulo_2pi(tangente[0], tangente[0], tangente[2], tangente[3])
        couple = [[super_arete[0], super_arete[1]], [super_arete[2], super_arete[3]]]
        mauvaises_extremité = [
            mauvaises_extremités[0],
            mauvaises_extremités[1],
            mauvaises_extremités[2],
            mauvaises_extremités[3],
        ]
        if (
            distance_modulo_2pi(tangente[0], tangente[0], tangente[1], tangente[3])
            < valeur
        ):
            valeur = distance_modulo_2pi(
                tangente[0], tangente[0], tangente[1], tangente[3]
            )
            couple = [
                [super_arete[0], super_arete[2]],
                [super_arete[1], super_arete[3]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[0],
                mauvaises_extremités[2],
                mauvaises_extremités[1],
                mauvaises_extremités[3],
            ]
        if (
            distance_modulo_2pi(tangente[0], tangente[0], tangente[1], tangente[2])
            < valeur
        ):
            valeur = distance_modulo_2pi(
                tangente[0], tangente[0], tangente[1], tangente[2]
            )
            couple = [
                [super_arete[0], super_arete[3]],
                [super_arete[1], super_arete[2]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[0],
                mauvaises_extremités[3],
                mauvaises_extremités[1],
                mauvaises_extremités[2],
            ]
        return [couple, mauvaises_extremité]
    elif len(super_arete[1].liste_des_points) == 1:
        valeur = distance_modulo_2pi(tangente[1], tangente[1], tangente[2], tangente[3])
        couple = [[super_arete[0], super_arete[1]], [super_arete[2], super_arete[3]]]
        mauvaises_extremité = [
            mauvaises_extremités[0],
            mauvaises_extremités[1],
            mauvaises_extremités[2],
            mauvaises_extremités[3],
        ]
        if (
            distance_modulo_2pi(tangente[1], tangente[1], tangente[0], tangente[3])
            < valeur
        ):
            valeur = distance_modulo_2pi(
                tangente[1], tangente[1], tangente[0], tangente[3]
            )
            couple = [
                [super_arete[1], super_arete[2]],
                [super_arete[0], super_arete[3]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[1],
                mauvaises_extremités[2],
                mauvaises_extremités[0],
                mauvaises_extremités[3],
            ]
        if (
            distance_modulo_2pi(tangente[0], tangente[0], tangente[0], tangente[2])
            < valeur
        ):
            valeur = distance_modulo_2pi(
                tangente[1], tangente[1], tangente[0], tangente[2]
            )
            couple = [
                [super_arete[1], super_arete[3]],
                [super_arete[0], super_arete[2]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[1],
                mauvaises_extremités[3],
                mauvaises_extremités[0],
                mauvaises_extremités[2],
            ]
        return [couple, mauvaises_extremité]
    elif len(super_arete[2].liste_des_points) == 1:
        valeur = distance_modulo_2pi(tangente[2], tangente[2], tangente[0], tangente[3])
        couple = [[super_arete[1], super_arete[2]], [super_arete[0], super_arete[3]]]
        mauvaises_extremité = [
            mauvaises_extremités[1],
            mauvaises_extremités[2],
            mauvaises_extremités[0],
            mauvaises_extremités[3],
        ]
        if (
            distance_modulo_2pi(tangente[2], tangente[2], tangente[1], tangente[3])
            < valeur
        ):
            valeur = distance_modulo_2pi(
                tangente[0], tangente[0], tangente[1], tangente[3]
            )
            couple = [
                [super_arete[0], super_arete[2]],
                [super_arete[1], super_arete[3]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[0],
                mauvaises_extremités[2],
                mauvaises_extremités[1],
                mauvaises_extremités[3],
            ]
        if (
            distance_modulo_2pi(tangente[2], tangente[2], tangente[0], tangente[1])
            < valeur
        ):
            valeur = distance_modulo_2pi(
                tangente[2], tangente[2], tangente[0], tangente[1]
            )
            couple = [
                [super_arete[2], super_arete[3]],
                [super_arete[0], super_arete[1]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[2],
                mauvaises_extremités[3],
                mauvaises_extremités[0],
                mauvaises_extremités[1],
            ]
        return [couple, mauvaises_extremité]
    elif len(super_arete[3].liste_des_points) == 1:
        valeur = distance_modulo_2pi(tangente[3], tangente[3], tangente[0], tangente[1])
        couple = [[super_arete[2], super_arete[3]], [super_arete[0], super_arete[1]]]
        mauvaises_extremité = [
            mauvaises_extremités[2],
            mauvaises_extremités[3],
            mauvaises_extremités[0],
            mauvaises_extremités[1],
        ]
        if (
            distance_modulo_2pi(tangente[3], tangente[3], tangente[1], tangente[2])
            < valeur
        ):
            valeur = distance_modulo_2pi(
                tangente[3], tangente[3], tangente[1], tangente[2]
            )
            couple = [
                [super_arete[0], super_arete[3]],
                [super_arete[1], super_arete[2]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[0],
                mauvaises_extremités[3],
                mauvaises_extremités[1],
                mauvaises_extremités[2],
            ]
        if (
            distance_modulo_2pi(tangente[3], tangente[3], tangente[0], tangente[2])
            < valeur
        ):
            valeur = distance_modulo_2pi(
                tangente[3], tangente[3], tangente[0], tangente[2]
            )
            couple = [
                [super_arete[1], super_arete[3]],
                [super_arete[0], super_arete[2]],
            ]
            mauvaises_extremité = [
                mauvaises_extremités[1],
                mauvaises_extremités[3],
                mauvaises_extremités[0],
                mauvaises_extremités[2],
            ]
        return [couple, mauvaises_extremité]
    else:
        return [couple, mauvaises_extremités]


def occurence(tableau, nombre):
    n = len(tableau)
    compteur = 0
    for i in range(0, n):
        if tableau[i] == nombre:
            compteur = compteur + 1
    return compteur


def test_si_une_arete_est_de_taille_1_3_aretes(
    liste_des_noeuds, tangente, mauvaises_extremités, super_arete, courbure
):
    [couple, mauvaises_extremité, valeurs, courbure_0] = [[], [], [], []]
    taille_0 = len(super_arete[0].liste_des_points)
    taille_1 = len(super_arete[1].liste_des_points)
    taille_2 = len(super_arete[2].liste_des_points)
    taille = [taille_0, taille_1, taille_2]
    nombre_de_1 = occurence(taille, 1)
    if nombre_de_1 == 0:
        return [couple, mauvaises_extremité, valeurs, courbure_0]
    elif nombre_de_1 == 1:
        if len(super_arete[0].liste_des_points) == 1:
            if (
                compter_le_nombre_de_noeud_pour_une_arete(
                    super_arete[0], liste_des_noeuds
                )
                == 1
                and critere_de_courbure(courbure[1], courbure[2])
                and distance_modulo_2pi_deux_valeurs(tangente[1], tangente[2])
                < parametre_trois_aretes_un_point
            ):
                couple.append([super_arete[1], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[1], mauvaises_extremités[2]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[1], courbure[2]])
            elif (
                distance_modulo_2pi_deux_valeurs(tangente[1], tangente[2])
                < parametre_trois_aretes_un_point
            ):
                couple.append([super_arete[1], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[1], mauvaises_extremités[2]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[1], courbure[2]])
            else:
                couple.append([super_arete[0], super_arete[1]])
                couple.append([super_arete[0], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[1]]
                )
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[2]]
                )
                valeurs.append(0)
                valeurs.append(0)
                courbure_0.append([courbure[0], courbure[1]])
                courbure_0.append([courbure[0], courbure[2]])
        if len(super_arete[1].liste_des_points) == 1:
            if (
                compter_le_nombre_de_noeud_pour_une_arete(
                    super_arete[1], liste_des_noeuds
                )
                == 1
                and critere_de_courbure(courbure[0], courbure[2])
                and distance_modulo_2pi_deux_valeurs(tangente[0], tangente[2])
                < parametre_trois_aretes_un_point
            ):
                couple.append([super_arete[0], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[2]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[0], courbure[2]])
            elif (
                distance_modulo_2pi_deux_valeurs(tangente[0], tangente[2])
                < parametre_trois_aretes_un_point
            ):
                couple.append([super_arete[0], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[2]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[0], courbure[2]])
            else:
                couple.append([super_arete[1], super_arete[0]])
                couple.append([super_arete[1], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[1], mauvaises_extremités[0]]
                )
                mauvaises_extremité.append(
                    [mauvaises_extremités[1], mauvaises_extremités[2]]
                )
                valeurs.append(0)
                valeurs.append(0)
                courbure_0.append([courbure[1], courbure[0]])
                courbure_0.append([courbure[1], courbure[2]])
        if len(super_arete[2].liste_des_points) == 1:
            if (
                compter_le_nombre_de_noeud_pour_une_arete(
                    super_arete[2], liste_des_noeuds
                )
                == 1
                and critere_de_courbure(courbure[0], courbure[1])
                and distance_modulo_2pi_deux_valeurs(tangente[0], tangente[1])
                < parametre_trois_aretes_un_point
            ):
                couple.append([super_arete[0], super_arete[1]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[1]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[0], courbure[1]])
            elif (
                distance_modulo_2pi_deux_valeurs(tangente[0], tangente[1])
                < parametre_trois_aretes_un_point
            ):
                couple.append([super_arete[0], super_arete[1]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[1]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[0], courbure[1]])
            else:
                couple.append([super_arete[2], super_arete[0]])
                couple.append([super_arete[2], super_arete[1]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[2], mauvaises_extremités[0]]
                )
                mauvaises_extremité.append(
                    [mauvaises_extremités[2], mauvaises_extremités[1]]
                )
                valeurs.append(0)
                valeurs.append(0)
                courbure_0.append([courbure[2], courbure[0]])
                courbure_0.append([courbure[2], courbure[1]])
        return [couple, mauvaises_extremité, valeurs, courbure_0]
    elif nombre_de_1 == 2:
        if taille_0 != 1:
            dist_0 = distance_modulo_2pi_deux_valeurs(
                tangente[0], angle(mauvaises_extremités[1], mauvaises_extremités[0])
            )
            dist_1 = distance_modulo_2pi_deux_valeurs(
                tangente[0], angle(mauvaises_extremités[2], mauvaises_extremités[0])
            )
            if dist_0 < dist_1:
                couple.append([super_arete[0], super_arete[1]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[1]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[0], courbure[1]])
            else:
                couple.append([super_arete[0], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[0], mauvaises_extremités[2]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[0], courbure[2]])
        if taille_1 != 1:
            dist_0 = distance_modulo_2pi_deux_valeurs(
                tangente[1], angle(mauvaises_extremités[0], mauvaises_extremités[1])
            )
            dist_1 = distance_modulo_2pi_deux_valeurs(
                tangente[1], angle(mauvaises_extremités[2], mauvaises_extremités[1])
            )
            if dist_0 < dist_1:
                couple.append([super_arete[1], super_arete[0]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[1], mauvaises_extremités[0]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[1], courbure[0]])
            else:
                couple.append([super_arete[1], super_arete[2]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[1], mauvaises_extremités[2]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[1], courbure[2]])
        if taille_2 != 1:
            dist_0 = distance_modulo_2pi_deux_valeurs(
                tangente[2], angle(mauvaises_extremités[0], mauvaises_extremités[2])
            )
            dist_1 = distance_modulo_2pi_deux_valeurs(
                tangente[2], angle(mauvaises_extremités[1], mauvaises_extremités[2])
            )
            if dist_0 < dist_1:
                couple.append([super_arete[2], super_arete[0]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[2], mauvaises_extremités[0]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[2], courbure[0]])
            else:
                couple.append([super_arete[2], super_arete[1]])
                mauvaises_extremité.append(
                    [mauvaises_extremités[2], mauvaises_extremités[1]]
                )
                valeurs.append(0)
                courbure_0.append([courbure[2], courbure[1]])
        return [couple, mauvaises_extremité, valeurs, courbure_0]
    elif nombre_de_1 == 3:
        return [couple, mauvaises_extremité, valeurs, courbure_0]


def renvoie_les_tangentes_et_les_mauvaises_extremités(noeud):
    n = len(noeud.liste_des_super_aretes_connectées)
    tangente = []
    mauvaises_extremités = []
    super_aretes = []
    courbure = []
    for i in range(0, n):
        super_arete = noeud.liste_des_super_aretes_connectées[i]
        super_aretes.append(super_arete)
        if noeud.extremités[i][0]:
            tangente.append(super_arete.tangente[0])
            mauvaises_extremités.append(super_arete.extremités[0])
            courbure.append(super_arete.courbure[0])
        else:
            tangente.append(super_arete.tangente[1])
            mauvaises_extremités.append(super_arete.extremités[1])
            courbure.append(super_arete.courbure[1])
    return [tangente, mauvaises_extremités, super_aretes, courbure]


def renvoie_la_courbure_et_les_mauvaises_extremités(noeud):
    n = len(noeud.liste_des_super_aretes_connectées)
    courbure = []
    mauvaises_extremités = []
    super_aretes = []
    for i in range(0, n):
        super_arete = noeud.liste_des_super_aretes_connectées[i]
        super_aretes.append(super_arete)
        if noeud.extremités[i][0]:
            courbure.append(super_arete.courbure[0])
            mauvaises_extremités.append(super_arete.extremités[0])
        else:
            courbure.append(super_arete.courbure[1])
            mauvaises_extremités.append(super_arete.extremités[1])
    return [courbure, mauvaises_extremités, super_aretes]


def trouver_des_liens_4_aretes(liste_des_noeuds, liste_des_aretes, indice_de_confiance):
    n = len(liste_des_noeuds)
    k = 0
    liste_des_mauvais_choix_arete = []
    liste_des_mauvais_choix_noeuds = []
    while k < nombre_iteration and n != 0:
        k = k + 1
        i = random.randint(0, n - 1)
        noeud = liste_des_noeuds[i]
        if len(noeud.liste_des_super_aretes_connectées) == 4:
            couple_general = couple_deux_a_deux_tangente(noeud, indice_de_confiance)
            confiance = couple_general[2]
            if confiance > indice_de_confiance:
                couple = couple_general[0]
                mauvaise_extremités = couple_general[1]
                fusion_1 = fusion_deux_super_aretes(
                    couple[0][0],
                    couple[0][1],
                    mauvaise_extremités[0],
                    mauvaise_extremités[1],
                )
                fusion_2 = fusion_deux_super_aretes(
                    couple[1][0],
                    couple[1][1],
                    mauvaise_extremités[2],
                    mauvaise_extremités[3],
                )
                if (
                    fusion_1.correlation_courbure < 0.7
                    and fusion_1.correlation_courbure != 0
                    and len(fusion_1.liste_des_points) > 40
                ) or (
                    fusion_2.correlation_courbure < 0.7
                    and fusion_2.correlation_courbure != 0
                    and len(fusion_2.liste_des_points) > 40
                ):
                    liste_des_mauvais_choix_arete.append(couple[0][0])
                    liste_des_mauvais_choix_arete.append(couple[0][1])
                    liste_des_mauvais_choix_arete.append(couple[1][0])
                    liste_des_mauvais_choix_arete.append(couple[1][1])
                    liste_des_mauvais_choix_noeuds.append(noeud)
                    retirer_un_element_liste_de_super_aretes(
                        couple[0][0], liste_des_aretes
                    )
                    retirer_un_element_liste_de_super_aretes(
                        couple[0][1], liste_des_aretes
                    )
                    retirer_un_element_liste_de_super_aretes(
                        couple[1][0], liste_des_aretes
                    )
                    retirer_un_element_liste_de_super_aretes(
                        couple[1][1], liste_des_aretes
                    )
                    implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)
                    affichage_features_on_image(image, fusion_1.liste_des_points)
                    affichage_features_on_image(image, fusion_2.liste_des_points)
                else:
                    retirer_un_element_liste_de_super_aretes(
                        couple[0][0], liste_des_aretes
                    )
                    retirer_un_element_liste_de_super_aretes(
                        couple[0][1], liste_des_aretes
                    )
                    retirer_un_element_liste_de_super_aretes(
                        couple[1][0], liste_des_aretes
                    )
                    retirer_un_element_liste_de_super_aretes(
                        couple[1][1], liste_des_aretes
                    )
                    ajouter_un_element_liste_de_super_aretes(fusion_1, liste_des_aretes)
                    ajouter_un_element_liste_de_super_aretes(fusion_2, liste_des_aretes)
                    retirer_un_element_liste_de_super_noeuds(noeud, liste_des_noeuds)
                    implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)
        n = len(liste_des_noeuds)
    for p in liste_des_mauvais_choix_arete:
        ajouter_un_element_liste_de_super_aretes(p, liste_des_aretes)
    implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)
    return ()


def trouver_des_liens_3_aretes(liste_des_noeuds, liste_des_aretes, indice_de_confiance):
    noeud = trouver_le_bon_noeud_avec_3_aretes(liste_des_noeuds)
    k = 0
    lst = []
    while k < nombre_iteration and not (super_noeuds_egaux(noeud, Super_noeud())):
        couple_general = couple_pour_3_aretes(
            liste_des_noeuds, noeud, indice_de_confiance
        )
        couple = couple_general[0]
        m = len(couple)
        if m != 0:
            for i in range(0, m):
                mauvaise_extremités = couple_general[1]
                fusion_1 = fusion_deux_super_aretes(
                    couple[i][0],
                    couple[i][1],
                    mauvaise_extremités[i][0],
                    mauvaise_extremités[i][1],
                )

                retirer_un_element_liste_de_super_aretes(couple[i][0], liste_des_aretes)
                retirer_un_element_liste_de_super_aretes(couple[i][1], liste_des_aretes)
                distance = np.sqrt(
                    (fusion_1.extremités[0][0] - fusion_1.extremités[1][0]) ** 2
                    + (fusion_1.extremités[0][1] - fusion_1.extremités[1][1]) ** 2
                )
                if len(fusion_1.liste_des_points) > 5 and (
                    (8 * distance) < len(fusion_1.liste_des_points)
                ):
                    new_cercle = Cercle()
                    p = len(fusion_1.liste_des_points)
                    raccord = segment_fusion(
                        fusion_1.liste_des_points[p - 1], fusion_1.liste_des_points[0]
                    )
                    liste_des_points_cercle = fusion_1.liste_des_points
                    liste_des_points_cercle = liste_des_points_cercle + raccord
                    liste_des_points_cercle = enlever_doublon(liste_des_points_cercle)
                    new_cercle.liste_des_points = liste_des_points_cercle
                    liste_des_super_cercles.append(new_cercle)
                else:
                    ajouter_un_element_liste_de_super_aretes(fusion_1, liste_des_aretes)
        if m == 0:
            lst.append(noeud)
        retirer_un_element_liste_de_super_noeuds(noeud, liste_des_noeuds)
        implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)
        noeud = trouver_le_bon_noeud_avec_3_aretes(liste_des_noeuds)
        k = k + 1
    for p in lst:
        ajouter_un_element_liste_de_super_noeuds(p, liste_des_noeuds)
    implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)
    return ()


def trouver_des_liens_2_aretes(liste_des_noeuds, liste_des_aretes, indice_de_confiance):
    noeud = trouver_le_bon_noeud_avec_2_aretes(liste_des_noeuds)
    k = 0
    while k < nombre_iteration and not (super_noeuds_egaux(noeud, Super_noeud())):
        couple_general = couple_pour_2_aretes(
            liste_des_noeuds, noeud, indice_de_confiance
        )
        couple = couple_general[0]
        m = len(couple)
        for i in range(0, m):
            mauvaise_extremités = couple_general[1]
            fusion_1 = fusion_deux_super_aretes(
                couple[i][0],
                couple[i][1],
                mauvaise_extremités[i][0],
                mauvaise_extremités[i][1],
            )
            if (
                fusion_1.correlation_courbure < 0.7
                and fusion_1.correlation_courbure != 0
                and len(fusion_1.liste_des_points) > 40
            ):
                liste_des_mauvais_choix.append(fusion_1)
                retirer_un_element_liste_de_super_aretes(couple[i][0], liste_des_aretes)
                retirer_un_element_liste_de_super_aretes(couple[i][1], liste_des_aretes)
            else:
                retirer_un_element_liste_de_super_aretes(couple[i][0], liste_des_aretes)
                retirer_un_element_liste_de_super_aretes(couple[i][1], liste_des_aretes)
                ajouter_un_element_liste_de_super_aretes(fusion_1, liste_des_aretes)
        retirer_un_element_liste_de_super_noeuds(noeud, liste_des_noeuds)
        implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)
        noeud = trouver_le_bon_noeud_avec_2_aretes(liste_des_noeuds)
        k = k + 1
    return ()


def trouver_le_bon_noeud_avec_3_aretes(liste_des_noeuds):
    noeud_0 = Super_noeud()
    compteur = 0
    for noeud in liste_des_noeuds:
        if noeud.nombre_aretes == 3:
            compteur_1 = min(
                len(noeud.liste_des_super_aretes_connectées[0].liste_des_points),
                len(noeud.liste_des_super_aretes_connectées[1].liste_des_points),
                len(noeud.liste_des_super_aretes_connectées[2].liste_des_points),
            )
            if compteur_1 >= compteur:
                noeud_0 = noeud
                compteur = compteur_1
    return noeud_0


def trouver_le_bon_noeud_avec_2_aretes(liste_des_noeuds):
    noeud_0 = Super_noeud()
    compteur = 0
    if liste_des_noeuds != []:
        for noeud in liste_des_noeuds:
            if noeud.nombre_aretes == 2:
                compteur_1 = min(
                    len(noeud.liste_des_super_aretes_connectées[0].liste_des_points),
                    len(noeud.liste_des_super_aretes_connectées[1].liste_des_points),
                )
                if compteur_1 >= compteur:
                    noeud_0 = noeud
                    compteur = compteur_1
        return noeud_0
    else:
        return Super_noeud()


def trouver_le_bon_noeud_avec_4_aretes(liste_des_noeuds):
    noeud_0 = Super_noeud()
    compteur = 0
    for noeud in liste_des_noeuds:
        if noeud.nombre_aretes == 4:
            compteur_1 = min(
                len(noeud.liste_des_super_aretes_connectées[0].liste_des_points),
                len(noeud.liste_des_super_aretes_connectées[1].liste_des_points),
                len(noeud.liste_des_super_aretes_connectées[2].liste_des_points),
                len(noeud.liste_des_super_aretes_connectées[3].liste_des_points),
            )
            if compteur_1 >= compteur:
                noeud_0 = noeud
                compteur = compteur_1
    return noeud_0


def segment_fusion(point_1, point_2):
    segment_0 = []
    compteur = 0
    if point_1[0] == point_2[0]:
        if point_1[1] < point_2[1]:
            for i in range(point_1[1], point_2[1] + 1):
                segment_0.append([point_1[0], i])
                compteur = compteur + 1
        else:
            for i in range(point_2[1], point_1[1] + 1):
                segment_0.append([point_1[0], i])
                compteur = compteur + 1
            if compteur != 1:
                segment_0.reverse()
    elif point_1[1] == point_2[1]:
        if point_1[0] < point_2[0]:
            for i in range(point_1[0], point_2[0] + 1):
                segment_0.append([i, point_1[1]])
                compteur = compteur + 1
        else:
            for i in range(point_2[0], point_1[0] + 1):
                segment_0.append([i, point_1[1]])
                compteur = compteur + 1
            if compteur != 1:
                segment_0.reverse()
    else:
        if point_2[0] > point_1[0]:
            theta = np.arctan((point_1[1] - point_2[1]) / (point_1[0] - point_2[0]))
        else:
            theta = (
                np.arctan((point_1[1] - point_2[1]) / (point_1[0] - point_2[0])) + np.pi
            )
        if (theta <= np.pi / 4 and theta >= -np.pi / 4) or (
            theta <= 5 * np.pi / 4 and theta >= 3 * np.pi / 4
        ):
            for x in range(
                min(point_1[0], point_2[0]), max(point_1[0], point_2[0]) + 1
            ):
                f = int(np.tan(theta) * (x - point_1[0]) + point_1[1])
                if f >= min(point_1[1], point_2[1]) and f <= max(
                    point_1[1], point_2[1]
                ):
                    segment_0.append([x, f])
                    compteur = compteur + 1
            if point_1[0] > point_2[0] and compteur != 1:
                segment_0.reverse()
        else:
            for y in range(
                min(point_1[1], point_2[1]), max(point_1[1], point_2[1]) + 1
            ):
                x = int(1 / np.tan(theta) * (y - point_1[1]) + point_1[0])
                if x >= min(point_1[0], point_2[0]) and x <= max(
                    point_1[0], point_2[0]
                ):
                    segment_0.append([x, y])
                    compteur = compteur + 1
            if point_1[1] > point_2[1] and compteur != 1:
                segment_0.reverse()
    return segment_0


def supprimer_arete_taille_1_liée_a_aucun_noeud(liste_des_aretes, liste_des_noeuds):
    for arete in liste_des_aretes:
        accumulateur = compter_le_nombre_de_noeud_pour_une_arete(
            arete, liste_des_noeuds
        )
        if accumulateur == 0 and arete.longueur < 5:
            retirer_un_element_liste_de_super_aretes(arete, liste_des_aretes)
    implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)
    return ()


def compter_le_nombre_de_noeud_pour_une_arete(super_arete, liste_des_noeuds):
    accumulateur = 0
    for noeud in liste_des_noeuds:
        if super_arete in noeud.liste_des_super_aretes_connectées:
            accumulateur = accumulateur + 1
    return accumulateur


def fusion_deux_super_aretes(
    super_arete_1, super_arete_2, mauvaise_extremité_1, mauvaise_extremité_2
):
    fusion_super_aretes = Super_arete()
    nouvelle_liste_de_points = []
    fusion_extremités = [[0.0, 0.0], [0.0, 0.0]]
    if super_arete_1.extremités[0] != mauvaise_extremité_1:
        nouvelle_liste_de_points = (
            nouvelle_liste_de_points + super_arete_1.liste_des_points
        )
        fusion_extremités[0] = super_arete_1.extremités[0]
    else:
        oki = super_arete_1.liste_des_points
        oki.reverse()
        nouvelle_liste_de_points = nouvelle_liste_de_points + oki
        oki.reverse()
        fusion_extremités[0] = super_arete_1.extremités[1]
    raccord = segment_fusion(mauvaise_extremité_1, mauvaise_extremité_2)
    nouvelle_liste_de_points = nouvelle_liste_de_points + raccord
    if super_arete_2.extremités[1] != mauvaise_extremité_2:
        nouvelle_liste_de_points = (
            nouvelle_liste_de_points + super_arete_2.liste_des_points
        )
        fusion_extremités[1] = super_arete_2.extremités[1]
    else:
        okii = super_arete_2.liste_des_points
        okii.reverse()
        nouvelle_liste_de_points = nouvelle_liste_de_points + okii
        okii.reverse()
        fusion_extremités[1] = super_arete_2.extremités[0]
    nouvelle_liste_de_points = enlever_doublon(nouvelle_liste_de_points)
    fusion_super_aretes.liste_des_points = nouvelle_liste_de_points
    fusion_super_aretes.longueur = (
        super_arete_1.longueur + super_arete_2.longueur + len(raccord)
    )
    fusion_super_aretes.extremités = fusion_extremités
    fusion_super_aretes.angle = implementer_angle_super_aretes(fusion_super_aretes)
    implementer_regression_liste_super_aretes([fusion_super_aretes])
    implementer_tangente_liste_super_aretes([fusion_super_aretes])

    return fusion_super_aretes


def créer_des_cercles(liste_des_noeuds, liste_des_aretes):
    liste_des_cercles = []
    liste_tempo_arete = []
    liste_des_aretes_bien = []
    for noeud in liste_des_noeuds:
        n = len(noeud.extremités)
        liste_tempo = []
        liste_tempo_arete = []
        for i in range(0, n):
            super_arete = noeud.liste_des_super_aretes_connectées[i]
            m = len(super_arete.liste_des_points)
            if noeud.extremités[i][0] and noeud.extremités[i][1] and m > 10:
                new_cercle = Cercle()
                liste_des_points_cercle = super_arete.liste_des_points
                liste_des_points_cercle = enlever_doublon(liste_des_points_cercle)
                new_cercle.liste_des_points = liste_des_points_cercle
                liste_tempo.append(new_cercle)
                liste_tempo_arete.append(super_arete)
        liste_des_aretes_bien = liste_des_aretes_bien + liste_tempo_arete
        liste_des_cercles = liste_des_cercles + liste_tempo
    for arete in liste_des_aretes_bien:
        retirer_un_element_liste_de_super_aretes(arete, liste_des_aretes)
    implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)
    liste_tempo = []
    liste_tempo_arete = []
    for arete in liste_des_aretes:
        m = len(arete.liste_des_points)
        distance = np.sqrt(
            (arete.extremités[0][0] - arete.extremités[1][0]) ** 2
            + (arete.extremités[0][1] - arete.extremités[1][1]) ** 2
        )
        if distance < 2.0:
            if est_un_cercle(arete):
                new_cercle = Cercle()
                liste_des_points_cercle = arete.liste_des_points
                new_cercle.liste_des_points = liste_des_points_cercle
                liste_tempo.append(new_cercle)
                liste_tempo_arete.append(arete)
        elif rapport_libre_sur_courbe_fusion * distance < m and m > 10:
            new_cercle = Cercle()
            liste_des_points_cercle = arete.liste_des_points
            liste_des_points_cercle = enlever_doublon(liste_des_points_cercle)
            new_cercle.liste_des_points = liste_des_points_cercle
            liste_tempo.append(new_cercle)
            liste_tempo_arete.append(arete)
    liste_des_cercles = liste_des_cercles + liste_tempo
    for arete in liste_tempo_arete:
        retirer_un_element_liste_de_super_aretes(arete, liste_des_aretes)
    implementer_aretes_des_noeuds(liste_des_noeuds, liste_des_aretes)

    return liste_des_cercles


def concatener_liste_de_cercles(liste_de_cercles_1, liste_de_cercles_2):
    for cercle in liste_de_cercles_2:
        liste_de_cercles_1.append(cercle)
    return liste_de_cercles_1


def est_un_cercle(super_arete):
    boo = True
    for p in super_arete.liste_des_points:
        liste_des_voisins = edges_and_nodes_creation.liste_de_voisins(
            p, super_arete.liste_des_points
        )
        if len(liste_des_voisins) != 2:
            boo = False
    return boo


def affichage_cercles_dans_liste_de_cercles(im, liste_de_cercles):
    new = []
    for cercle in liste_de_cercles:
        new = new + cercle.liste_des_points
    affichage_features_on_image(im, new)
    return ()


def afficher_des_cercles(liste_des_aretes):
    k = 0
    for p in liste_des_aretes:
        n = len(p.liste_des_points)
        distance = np.sqrt(
            (p.extremités[0][0] - p.extremités[1][0]) ** 2
            + (p.extremités[0][1] - p.extremités[1][1]) ** 2
        )
        if n > longueur_de_la_courbe and coeff_ouverture_courbe * distance < n:
            affichage_features_on_image(image, p.liste_des_points)
            k = k + 1
            print()
    return ()


def afficher_des_cercles_sur_meme_image(im, liste_des_aretes):
    lst = []
    for p in liste_des_aretes:
        n = len(p.liste_des_points)
        distance = np.sqrt(
            (p.extremités[0][0] - p.extremités[1][0]) ** 2
            + (p.extremités[0][1] - p.extremités[1][1]) ** 2
        )
        if n > longueur_de_la_courbe and coeff_ouverture_courbe * distance < n:
            lst = lst + p.liste_des_points
    affichage_features_on_image(im, lst)
    print()
    return ()


def afficher_liste_de_noeuds(liste_des_noeuds):
    for p in liste_des_noeuds:
        affichage_features_on_image(image, p.liste_des_points)
        for arete in p.liste_des_super_aretes_connectées:
            affichage_features_on_image(image, arete.liste_des_points)
    return ()


affichage_features_on_image
print()
