import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import features_automatic_detection
import graph_class
import edges_and_nodes_creation


gaussian_blur = 13
threshold = 0.01

graph_class.indice_de_confiance_1 = 10
graph_class.indice_de_confiance_2 = 3
graph_class.indice_de_confiance_3 = 0.7
graph_class.indice_de_confiance_4 = (
    8
)  # plus il est elevé plus il est sélectif (quand il est inferieur a 1 il est non selectif) indice qui joue sur la curvature
graph_class.nombre_iteration = 100000
graph_class.parametre_trois_aretes_un_point = 0.1
graph_class.longueur_de_la_courbe = 4
graph_class.coeff_ouverture_courbe = 0.1
graph_class.rapport_libre_sur_courbe_fusion = 5

graph_class.param_courbure = 0.001
graph_class.param_angle = 0.01

graph_class.liste_des_super_cercles = []

edges_and_nodes_creation.couleur_de_affichage = -40


def moins(im):
    n = len(im)
    tab = np.zeros((n, n), dtype=float)
    for i in range(0, n):
        for j in range(0, n):
            tab[i, j] = -im[i, j]
    return tab


def moins_tableau(im):
    n = len(im)
    tab = np.zeros(n, dtype=float)
    for i in range(0, n):
        tab[i] = -im[i]
    return tab


def transformer_image_liste_en_tableau(liste):
    n = len(liste)
    p = int(np.sqrt(n))
    tableau_image = np.zeros((p, p), dtype=float)
    for i in range(0, p):
        for j in range(0, p):
            tableau_image[i, j] = liste[i * p + j]
    return tableau_image


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


def local_main_procedure(liste_image):
    features_automatic_detection.ecart_type_blur = gaussian_blur
    features_automatic_detection.parametre = threshold
    features_automatic_detection.largeur_max = int(np.sqrt(len(liste_image)))
    features_automatic_detection.hauteur_max = int(np.sqrt(len(liste_image)))
    image = transformer_image_liste_en_tableau(liste_image)
    image_features = features_automatic_detection.procedure_canny(image)
    image_skel = features_automatic_detection.tableau_features(image)
    print("basic_picture")
    print_matrix(image)
    print("picture_with_features")
    print_matrix(image_features)
    print("skeletonize_picture")
    print_matrix(image_skel)
    liste_skel_originale = edges_and_nodes_creation.coordinate_skel(image_skel)
    lst_skel = liste_skel_originale.copy()
    pixels_noeuds = edges_and_nodes_creation.retirer_les_noeuds(lst_skel)
    liste_des_aretes = edges_and_nodes_creation.liste_des_aretes(lst_skel)
    liste_des_noeuds = edges_and_nodes_creation.liste_des_noeuds(pixels_noeuds)
    image_avec_les_aretes = edges_and_nodes_creation.affichage_general(
        image, liste_des_aretes
    )
    image_avec_les_noeuds = edges_and_nodes_creation.affichage_general(
        image, liste_des_noeuds
    )
    print_matrix(image_avec_les_aretes)
    print_matrix(image_avec_les_noeuds)
    liste_de_super_aretes = []
    liste_de_super_noeuds = []
    liste_de_super_aretes = graph_class.implementer_une_liste_de_super_aretes(
        liste_des_aretes
    )
    liste_de_super_noeuds = graph_class.implementer_une_liste_de_super_noeuds(
        liste_des_noeuds
    )
    graph_class.implementer_extremites_liste_super_aretes(liste_de_super_aretes)
    graph_class.implementer_angle_liste_super_aretes(liste_de_super_aretes)
    graph_class.implementer_regression_liste_super_aretes(liste_de_super_aretes)
    graph_class.implementer_tangente_liste_super_aretes(liste_de_super_aretes)
    graph_class.implementer_aretes_des_noeuds(
        liste_de_super_noeuds, liste_de_super_aretes
    )
    graph_class.supprimer_arete_taille_1_liée_a_aucun_noeud(
        liste_de_super_aretes, liste_de_super_noeuds
    )

    print("etape_1")
    graph_class.trouver_des_liens_4_aretes(
        liste_de_super_noeuds, liste_de_super_aretes, graph_class.indice_de_confiance_1
    )
    liste_de_cercle = graph_class.créer_des_cercles(
        liste_de_super_noeuds, liste_de_super_aretes
    )

    print("etape_2")
    graph_class.trouver_des_liens_4_aretes(
        liste_de_super_noeuds, liste_de_super_aretes, graph_class.indice_de_confiance_2
    )
    liste_de_cercle_1 = graph_class.créer_des_cercles(
        liste_de_super_noeuds, liste_de_super_aretes
    )

    print("etape_3")
    graph_class.trouver_des_liens_3_aretes(
        liste_de_super_noeuds, liste_de_super_aretes, graph_class.indice_de_confiance_3
    )
    liste_de_cercle_2 = graph_class.créer_des_cercles(
        liste_de_super_noeuds, liste_de_super_aretes
    )

    print("etape_4")
    graph_class.trouver_des_liens_4_aretes(
        liste_de_super_noeuds, liste_de_super_aretes, graph_class.indice_de_confiance_2
    )
    liste_de_cercle_3 = graph_class.créer_des_cercles(
        liste_de_super_noeuds, liste_de_super_aretes
    )
    graph_class.trouver_des_liens_3_aretes(
        liste_de_super_noeuds, liste_de_super_aretes, graph_class.indice_de_confiance_3
    )

    print("etape_5")
    graph_class.trouver_des_liens_4_aretes(
        liste_de_super_noeuds, liste_de_super_aretes, graph_class.indice_de_confiance_2
    )
    graph_class.trouver_des_liens_2_aretes(
        liste_de_super_noeuds, liste_de_super_aretes, graph_class.param_angle
    )

    print("etape_6")
    liste_de_cercle_4 = graph_class.créer_des_cercles(
        liste_de_super_noeuds, liste_de_super_aretes
    )

    print("etape_7")
    graph_class.rapport_libre_sur_courbe_fusion = 2.7
    liste_de_cercle_5 = graph_class.créer_des_cercles(
        liste_de_super_noeuds, liste_de_super_aretes
    )

    print("etape_8")
    super_cercle_1 = graph_class.concatener_liste_de_cercles(
        graph_class.concatener_liste_de_cercles(
            graph_class.concatener_liste_de_cercles(
                graph_class.concatener_liste_de_cercles(
                    graph_class.concatener_liste_de_cercles(
                        liste_de_cercle, liste_de_cercle_1
                    ),
                    liste_de_cercle_2,
                ),
                liste_de_cercle_3,
            ),
            liste_de_cercle_4,
        ),
        liste_de_cercle_5,
    )
    print("etape_9")
    print("etape_10")
    superrr = graph_class.concatener_liste_de_cercles(
        graph_class.liste_des_super_cercles, super_cercle_1
    )

    pickle.dump(superrr, open("liste_de_super_aretes_coooool_18.p", "wb"))
    pickle.dump(liste_de_super_aretes, open("liste_de_super_aretes_cool_18.p", "wb"))
