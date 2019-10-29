from __future__ import division
import features_automatic_detection
import numpy as np
import os
import graph_class
import pickle
import matplotlib.pyplot as plt
import edges_and_nodes_creation
import random
import ellipse_polaire


confidence_coefficient = 0.8
number_of_iterations = 100000
number_of_appearances = 5


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


class Super_noeud:
    def __init__(self):
        self.liste_des_points = []
        self.liste_des_super_aretes_connectées = []
        self.extremités = []
        self.nombre_aretes = 0



class Cercle:
    def __init__(self):
        self.liste_des_points = []
        self.rayon = 0.0
        self.centre = [0.0, 0.0]


def afficher_des_elements(im, liste_des_aretes):
    lst = []
    for p in liste_des_aretes:
        lst = lst + p.liste_des_points
    graph_class.affichage_features_on_image(im, lst)
    print()
    return ()


def transformer_image_liste_en_tableau(liste):
    n = len(liste)
    p = int(np.sqrt(n))
    tableau_image = np.zeros((p, p), dtype=float)
    for i in range(0, p):
        for j in range(0, p):
            tableau_image[i, j] = liste[i * p + j]
    return tableau_image


features_automatic_detection.ecart_type_blur = 5
features_automatic_detection.parametre = 0.09
edges_and_nodes_creation.couleur_de_affichage = 250


def parametres_cercle(point_1, point_2, point_3):
    if (
        point_1[0] == point_2[0]
        or point_1[0] == point_3[0]
        or point_2[0] == point_3[0]
        or point_1[1] == point_2[1]
        or point_1[1] == point_3[1]
        or point_2[1] == point_3[1]
    ):
        return [0, 0, 0]
    else:
        if ((point_2[0] - point_1[0]) / (point_2[1] - point_1[1])) - (
            (point_3[0] - point_2[0]) / (point_3[1] - point_2[1])
        ) == 0:
            return [0, 0, 0]
        else:
            x_centre = -(
                (point_3[0] ** 2 - point_2[0] ** 2 + point_3[1] ** 2 - point_2[1] ** 2)
                / (2 * (point_3[1] - point_2[1]))
                - (
                    point_2[0] ** 2
                    - point_1[0] ** 2
                    + point_2[1] ** 2
                    - point_1[1] ** 2
                )
                / (2 * (point_2[1] - point_1[1]))
            ) / (
                ((point_2[0] - point_1[0]) / (point_2[1] - point_1[1]))
                - ((point_3[0] - point_2[0]) / (point_3[1] - point_2[1]))
            )
            y_centre = -(
                (point_2[0] - point_1[0]) / (point_2[1] - point_1[1])
            ) * x_centre + (
                point_2[0] ** 2 - point_1[0] ** 2 + point_2[1] ** 2 - point_1[1] ** 2
            ) / (
                2 * (point_2[1] - point_1[1])
            )
            rayon = np.sqrt((point_1[0] - x_centre) ** 2 + (point_1[1] - y_centre) ** 2)
            return [x_centre, y_centre, rayon]


def param_min(valeur_propre, vecteur_propre):
    lst = []
    n = len(valeur_propre)
    for i in range(n):
        lst.append(vecteur_propre[i][n - 1])
    return lst


def parametres_ellipse(point_1, point_2, point_3, point_4, point_5, point_6, point_7):
    vecteur_1 = [
        point_1[0] ** 2,
        2 * point_1[0] * point_1[1],
        point_1[1] ** 2,
        point_1[0] * 2,
        point_1[1] * 2,
        1,
    ]
    vecteur_2 = [
        point_2[0] ** 2,
        2 * point_2[0] * point_2[1],
        point_2[1] ** 2,
        point_2[0] * 2,
        point_2[1] * 2,
        1,
    ]
    vecteur_3 = [
        point_3[0] ** 2,
        2 * point_3[0] * point_3[1],
        point_3[1] ** 2,
        point_3[0] * 2,
        point_3[1] * 2,
        1,
    ]
    vecteur_4 = [
        point_4[0] ** 2,
        2 * point_4[0] * point_4[1],
        point_4[1] ** 2,
        point_4[0] * 2,
        point_4[1] * 2,
        1,
    ]
    vecteur_5 = [
        point_5[0] ** 2,
        2 * point_5[0] * point_5[1],
        point_5[1] ** 2,
        point_5[0] * 2,
        point_5[1] * 2,
        1,
    ]
    vecteur_6 = [
        point_6[0] ** 2,
        2 * point_6[0] * point_6[1],
        point_6[1] ** 2,
        point_6[0] * 2,
        point_6[1] * 2,
        1,
    ]
    vecteur_7 = [
        point_7[0] ** 2,
        2 * point_7[0] * point_7[1],
        point_7[1] ** 2,
        point_7[0] * 2,
        point_7[1] * 2,
        1,
    ]
    matrix = np.array(
        [vecteur_1, vecteur_2, vecteur_3, vecteur_4, vecteur_5, vecteur_6, vecteur_7]
    )
    oki = np.array(
        [vecteur_1, vecteur_2, vecteur_3, vecteur_4, vecteur_5, vecteur_6, vecteur_7]
    )
    impec = np.transpose(oki)
    symetrique = np.dot(impec, matrix)
    tab = np.linalg.eig(symetrique)
    valeur_propre = tab[0]
    vecteur_propre = tab[1]
    lst = param_min(valeur_propre, vecteur_propre)

    return lst


def ellipse(lst):
    tableau = np.zeros((256, 256), dtype=float)
    for x in range(256):
        for y in range(256):
            tableau[x][y] = abs(
                (
                    lst[0] * x ** 2
                    + lst[1] * 2 * x * y
                    + lst[2] * y ** 2
                    + lst[3] * x * 2
                    + lst[4] * y * 2
                    + lst[5]
                )
            )
    value = 10000 / (np.sqrt(abs(1 / (lst[0] + lst[2]))))
    liste = minimum(tableau, 1000000, value)
    return liste


def minimum(tab, coeff_1, nombre):
    lst = []
    for x in range(256):
        for y in range(256):
            if tab[x][y] < coeff_1:
                lst.append([x, y])
    n = len(lst)
    if n < nombre:
        return lst
    else:
        return minimum(tab, coeff_1 / 2, nombre)


def liste_cercle(a, b, r):
    lst = []
    for k in range(2000):
        x = int(a + r * np.cos(k * 2 * np.pi / 2000))
        y = int(b + r * np.sin(k * 2 * np.pi / 2000))
        if 0 <= x and 256 > x and 0 <= y and 256 > y and not ([x, y] in lst):
            lst.append([x, y])
    return lst


def comparaison_de_listes(cercle, image, param):
    acc = 0
    cercle = retirer_les_points_avec_trois_voisins(cercle)
    n = len(cercle)
    for p in cercle:
        if image[p[0], p[1]] != 0:
            acc = acc + 1
    if n != 0:
        coeff = acc / n
    else:
        coeff = 0
    if coeff > confidence_coefficient:
        print("Coeff", coeff)
        parametres_ellipse = ellipse_polaire.polaire_apres_rotation(param)
        if -np.arctan(parametres_ellipse[5] / parametres_ellipse[4]) < 0:
            ellipse_bonne_coordonnees = [
                parametres_ellipse[4] * parametres_ellipse[2]
                + parametres_ellipse[3] * parametres_ellipse[5],
                parametres_ellipse[4] * parametres_ellipse[3]
                - parametres_ellipse[5] * parametres_ellipse[2],
                parametres_ellipse[1],
                parametres_ellipse[0],
                -np.arctan(parametres_ellipse[5] / parametres_ellipse[4]) + np.pi / 2,
                coeff,
                cercle,
            ]
        else:
            ellipse_bonne_coordonnees = [
                parametres_ellipse[4] * parametres_ellipse[2]
                + parametres_ellipse[3] * parametres_ellipse[5],
                parametres_ellipse[4] * parametres_ellipse[3]
                - parametres_ellipse[5] * parametres_ellipse[2],
                parametres_ellipse[0],
                parametres_ellipse[1],
                -np.arctan(parametres_ellipse[5] / parametres_ellipse[4]),
                coeff,
                cercle,
            ]
        print("Centre : ", [ellipse_bonne_coordonnees[0], ellipse_bonne_coordonnees[1]])
        print("Rayon : ", [ellipse_bonne_coordonnees[2], ellipse_bonne_coordonnees[3]])
        print("Angle : ", [ellipse_bonne_coordonnees[4]])
        graph_class.affichage_features_on_image(image, ellipse_bonne_coordonnees[6])
        return ellipse_bonne_coordonnees
    else:
        return []


def comparaison_de_listes_cercles(cercle, image, param):
    acc = 0
    cercle = retirer_les_points_avec_trois_voisins(cercle)
    n = len(cercle)
    for p in cercle:
        if image[p[0], p[1]] != 0:
            acc = acc + 1
    if n != 0:
        coeff = acc / n
    else:
        coeff = 0
    parametres_ellipse = ellipse_polaire.polaire_apres_rotation(param)
    ellipse_bonne_coordonnees = [
        parametres_ellipse[4] * parametres_ellipse[2]
        + parametres_ellipse[3] * parametres_ellipse[5],
        parametres_ellipse[4] * parametres_ellipse[3]
        - parametres_ellipse[5] * parametres_ellipse[2],
        parametres_ellipse[0],
        parametres_ellipse[1],
        np.arccos(parametres_ellipse[4]),
        cercle,
    ]

    return [
        ellipse_bonne_coordonnees[0],
        ellipse_bonne_coordonnees[1],
        ellipse_bonne_coordonnees[2],
        ellipse_bonne_coordonnees[3],
        ellipse_bonne_coordonnees[4],
        coeff,
        ellipse_bonne_coordonnees[5],
    ]


def retirer_les_points_avec_trois_voisins(liste):
    for p in liste:
        voisin = edges_and_nodes_creation.liste_de_voisins(p, liste)
        if (
            ([p[0], p[1] + 1] in voisin and [p[0] + 1, p[1]] in voisin)
            or ([p[0], p[1] + 1] in voisin and [p[0] - 1, p[1]] in voisin)
            or ([p[0], p[1] - 1] in voisin and [p[0] - 1, p[1]] in voisin)
            or ([p[0], p[1] - 1] in voisin and [p[0] + 1, p[1]] in voisin)
        ):
            liste.remove(p)
    return liste


def transformer_en_tableau(liste):
    tab = np.zeros((256, 256))
    for p in liste:
        tab[p[0], p[1]] = 250
    return tab


def transformer_en_liste(tableau):
    liste = []
    for i in range(256):
        for j in range(256):
            if tableau[i, j] != 0:
                liste.append([i, j])
    return liste


def distance_euclidienne(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def test_plusieurs(k, image, liste_des_aretes_non_resolues):
    acc = []
    for i in range(k):
        n = len(liste_des_aretes_non_resolues)
        if n != 0:
            r = random.randint(0, n - 1)
            r_0 = random.randint(0, n - 1)
            p = len(liste_des_aretes_non_resolues[r].liste_des_points)
            m = len(liste_des_aretes_non_resolues[r_0].liste_des_points)
            if p > 30 and m > 4:
                test_1 = random.sample(
                    liste_des_aretes_non_resolues[r].liste_des_points, 4
                )
                test_2 = random.sample(
                    liste_des_aretes_non_resolues[r_0].liste_des_points, 3
                )
                if (
                    (test_1[0][0] - test_2[0][0]) ** 2
                    + (test_1[0][1] - test_2[0][1]) ** 2
                ) < 10000:
                    test = test_1 + test_2
                    param = parametres_ellipse(
                        [float(test[0][0]), float(test[0][1])],
                        [float(test[1][0]), float(test[1][1])],
                        [float(test[2][0]), float(test[2][1])],
                        [float(test[3][0]), float(test[3][1])],
                        [float(test[4][0]), float(test[4][1])],
                        [float(test[5][0]), float(test[5][1])],
                        [float(test[6][0]), float(test[6][1])],
                    )
                    if param[0] * param[2] > param[1] ** 2:
                        cercle = ellipse_polaire.ellipse_final(param)

                        oui = comparaison_de_listes(cercle, image, param)
                        if not (len(oui) == 0):
                            comparer_nouveau_cercle_liste(oui, acc)
    return acc


def test_plusieurs_cercles(k, image, liste_des_aretes_non_resolues):
    acc = []
    n = len(liste_des_aretes_non_resolues)
    if n != 0:
        for p in range(n):
            acc.append([])
        results = np.zeros(n)
        for i in range(k):
            r = random.randint(0, n - 1)
            p = len(liste_des_aretes_non_resolues[r].liste_des_points)
            if p > 10:
                test = random.sample(
                    liste_des_aretes_non_resolues[r].liste_des_points, 7
                )
                param = parametres_ellipse(
                    [float(test[0][0]), float(test[0][1])],
                    [float(test[1][0]), float(test[1][1])],
                    [float(test[2][0]), float(test[2][1])],
                    [float(test[3][0]), float(test[3][1])],
                    [float(test[4][0]), float(test[4][1])],
                    [float(test[5][0]), float(test[5][1])],
                    [float(test[6][0]), float(test[6][1])],
                )
                if param[0] * param[2] > param[1] ** 2:
                    cercle = ellipse_polaire.ellipse_final(param)

                    oui = comparaison_de_listes_cercles(cercle, image, param)
                    if oui[5] > results[r]:
                        results[r] = oui[5]
                        acc[r] = oui
    print(results)
    return acc


def compactifie(liste):
    accu = []
    for p in liste:
        if len(p[6]) != 0:
            accu = accu + p[6]
    return accu


def compactifie_nouveau(liste):
    accu = []
    for p in liste:
        if (
            (p[1] > number_of_appearances or p[0][5] > 0.87)
            and p[0][2] > 3
            and p[0][3] > 3
        ):
            accu = accu + p[0][6]
    return accu


def liste_superrr(liste_aretes):
    lst = []
    for p in liste_aretes:
        lst = lst + p.liste_des_points
    return lst


def metrique(lst_1, lst_2):
    [centre_x_1, centre_y_1, rayon_x_1, rayon_y_1, angle_1] = lst_1
    [centre_x_2, centre_y_2, rayon_x_2, rayon_y_2, angle_2] = lst_2
    if abs(angle_1 - angle_2) < np.pi / 4:
        d_angle = abs(angle_1 - angle_2)
        d_centre = np.sqrt(
            (centre_x_1 - centre_x_2) ** 2 + (centre_y_1 - centre_y_2) ** 2
        )
        d_rayon_x = abs((rayon_x_1 - rayon_x_2) / rayon_x_1)
        d_rayon_y = abs((rayon_y_1 - rayon_y_2) / rayon_y_1)
    else:
        if angle_1 > angle_2:
            d_angle = abs(angle_1 - np.pi / 2 - angle_2)
            d_centre = np.sqrt(
                (centre_x_1 - centre_x_2) ** 2 + (centre_y_1 - centre_y_2) ** 2
            )
            d_rayon_x = abs((rayon_x_1 - rayon_y_2) / rayon_x_1)
            d_rayon_y = abs((rayon_y_1 - rayon_x_2) / rayon_y_1)
        else:
            d_angle = abs(angle_2 - np.pi / 2 - angle_1)
            d_centre = np.sqrt(
                (centre_x_1 - centre_x_2) ** 2 + (centre_y_1 - centre_y_2) ** 2
            )
            d_rayon_x = abs((rayon_x_1 - rayon_y_2) / rayon_x_1)
            d_rayon_y = abs((rayon_y_1 - rayon_x_2) / rayon_y_1)
    return [d_angle, d_centre, d_rayon_x, d_rayon_y]


def est_le_meme_cercle(lst_1, lst_2):
    [d_angle, d_centre, d_rayon_x, d_rayon_y] = metrique(lst_1, lst_2)
    if d_angle < 0.3 and d_centre < 6 and d_rayon_x < 3 and d_rayon_y < 3:
        return True
    else:
        return False


def est_le_meme_cercle_algo_final(lst_1, lst_2):
    [d_angle, d_centre, d_rayon_x, d_rayon_y] = metrique(lst_1, lst_2)
    if d_centre < 7 and d_rayon_x < 5 and d_rayon_y < 5:
        return True
    else:
        return False


def comparer_nouveau_cercle_liste(lst, acc):
    compteur = 0
    for p in acc:
        if est_le_meme_cercle(p[0][0:5], lst[0:5]) and compteur == 0:
            if p[0][5] < lst[5]:
                p[0] = lst
            p[1] = actualiser(lst, p[1])
            print("act", p[1])
            compteur = 1
    if compteur == 0:
        acc.append([lst, 1])
    print(compteur)
    return ()


def actualiser(lst, info):
    info = info + 1
    return info


def fusion(lst_1, lst_2):
    new_lst = [0, 1]
    if est_le_meme_cercle_algo_final(lst_1[0][0:5], lst_2[0][0:5]):
        if lst_1[0][5] > lst_2[0][5]:
            new_lst[0] = lst_1[0]
        else:
            new_lst[0] = lst_2[0]
        new_lst[1] = lst_1[1] + lst_2[1]
    return new_lst


def regrouper_les_memes_cerles(lst_cercle):
    new_lst_cercle = []
    if len(lst_cercle) == 0:
        return []
    else:
        print(lst_cercle[0])
        new_lst_cercle.append(lst_cercle[0])
        n = len(lst_cercle)
        for p in lst_cercle[1 : n + 1]:
            compteur = 0
            for i in new_lst_cercle:
                if est_le_meme_cercle_algo_final(p[0][0:5], i[0][0:5]):
                    i = fusion(p, i)
                    compteur = 1
            if compteur == 0:
                new_lst_cercle.append(p)
        return new_lst_cercle


def print_matrix(matrix):
    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(8, 8),
        sharex=False,
        sharey=False,
        subplot_kw={"adjustable": "box-forced"},
    )
    ax1.imshow(matrix)
    plt.show()


def global_main_procedure(liste_image):
    image = transformer_image_liste_en_tableau(liste_image)
    liste_des_aretes_non_resolues = pickle.load(
        open("liste_de_super_aretes_cool_18.p", "rb")
    )
    liste_des_aretes_resolues = pickle.load(
        open("liste_de_super_aretes_coooool_18.p", "rb")
    )
    edges_and_nodes_creation.couleur_de_affichage = 350
    features_automatic_detection.largeur_max = int(np.sqrt(len(liste_image)))
    features_automatic_detection.hauteur_max = int(np.sqrt(len(liste_image)))
    image = transformer_image_liste_en_tableau(liste_image)
    image_features = features_automatic_detection.procedure_canny(image)

    if not (len(liste_des_aretes_resolues) == 0):
        ellipses_which_fit_well_local = test_plusieurs_cercles(
            1000, image_features, liste_des_aretes_resolues
        )
    else:
        ellipses_which_fit_well_local = []
    ellipses_which_fit_well_global = test_plusieurs(
        number_of_iterations, image_features, liste_des_aretes_non_resolues
    )
    print("ellipses_which_fit_well", ellipses_which_fit_well_global)
    print()

    edges_and_nodes_creation.couleur_de_affichage = -40
    liste_des_ellipses_sans_doublon_global = regrouper_les_memes_cerles(
        ellipses_which_fit_well_global
    )
    final_list_global = []
    for q in liste_des_ellipses_sans_doublon_global:
        final_list_global.append(q[0])
    final_list = final_list_global + ellipses_which_fit_well_local
    liste_des_points_des_ellipses_pour_laffichage_global = compactifie_nouveau(
        liste_des_ellipses_sans_doublon_global
    )
    liste_des_points_des_ellipses_pour_laffichage_local = compactifie(
        ellipses_which_fit_well_local
    )
    print("ellipses_detected_with_the_local_method")
    graph_class.affichage_features_on_image(
        image, liste_des_points_des_ellipses_pour_laffichage_local
    )
    print("ellipses_detected_with_the_global_method")
    graph_class.affichage_features_on_image(
        image, liste_des_points_des_ellipses_pour_laffichage_global
    )
    print("all_the_ellipses_detected")
    graph_class.affichage_features_on_image(
        image,
        liste_des_points_des_ellipses_pour_laffichage_global
        + liste_des_points_des_ellipses_pour_laffichage_local,
    )
    print("final_list", final_list)
