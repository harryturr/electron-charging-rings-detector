from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import features_automatic_detection

nombre_angle = 1000
features_automatic_detection.largeur_max = 256
features_automatic_detection.hauteur_max = 256


def ellipse_conversion(liste):
    coeff_poly_0 = -4 * liste[1] ** 2 - (liste[0] - liste[2]) ** 2
    coeff_poly_1 = (liste[0] - liste[2]) ** 2 + 4 * liste[1] ** 2
    coeff_poly_2 = -liste[1] ** 2
    cool = np.roots([coeff_poly_0, coeff_poly_1, coeff_poly_2])
    bons_beta = []
    if len(cool) == 0:
        return [1, 0]
    else:
        if cool[0] >= 0:
            beta_reel_0 = np.sqrt(cool[0])
            beta_reel_1 = -np.sqrt(cool[0])
            bons_beta.append(beta_reel_0)
            bons_beta.append(beta_reel_1)
        if cool[1] >= 0:
            beta_reel_2 = np.sqrt(cool[1])
            beta_reel_3 = -np.sqrt(cool[1])
            bons_beta.append(beta_reel_2)
            bons_beta.append(beta_reel_3)
        index = bons_beta[0]
        absol = abs(
            (liste[0] - liste[2]) * np.sqrt(1 - bons_beta[0] ** 2) * bons_beta[0]
            + liste[1] * (1 - 2 * bons_beta[0] ** 2)
        )
        n = len(bons_beta)
        for p in range(n):
            if abs(bons_beta[p]):
                value = (liste[0] - liste[2]) * np.sqrt(
                    1 - bons_beta[p] ** 2
                ) * bons_beta[p] + liste[1] * (1 - 2 * bons_beta[p] ** 2)
                absolu = abs(value)
                if absolu < absol:
                    absol = absolu
                    index = bons_beta[p]
        return [np.sqrt(1 - index ** 2), index]


def changement_de_coordonnées_ellipse(liste):
    [alpha, beta] = ellipse_conversion(liste)
    new_a = liste[0] * alpha ** 2 + liste[2] * beta ** 2 - 2 * liste[1] * alpha * beta
    new_b = 0
    new_c = liste[2] * alpha ** 2 + liste[0] * beta ** 2 + 2 * liste[1] * alpha * beta
    new_d = liste[3] * alpha - liste[4] * beta
    new_e = liste[4] * alpha + liste[3] * beta
    new_f = liste[5]
    new_coordonnees = [new_a, new_b, new_c, new_d, new_e, new_f]
    return new_coordonnees


def polaire_apres_rotation(liste):
    new = changement_de_coordonnées_ellipse(liste)
    K = new[3] ** 2 / new[0] + new[4] ** 2 / new[2] - new[5]
    rayon_x = np.sqrt(K / new[0])
    rayon_y = np.sqrt(K / new[2])
    centre_x = -new[3] / new[0]
    centre_y = -new[4] / new[2]
    [alpha, beta] = ellipse_conversion(liste)
    ellipse_polaire = [rayon_x, rayon_y, centre_x, centre_y, alpha, beta]
    return ellipse_polaire


def tracer_ellipse(liste):
    [rayon_x, rayon_y, centre_x, centre_y, alpha, beta] = polaire_apres_rotation(liste)
    liste = []
    for t in range(nombre_angle):
        cos = np.cos(2 * np.pi * t / nombre_angle)
        sin = np.sin(2 * np.pi * t / nombre_angle)
        [grand_X, grand_Y] = [centre_x + rayon_x * cos, centre_y + rayon_y * sin]
        [x, y] = [alpha * grand_X + beta * grand_Y, -beta * grand_X + alpha * grand_Y]
        [int_x, int_y] = [int(x), int(y)]
        if int_x >= 0 and int_x < 256 and int_y >= 0 and int_y < 256:
            liste.append([int_x, int_y])
    return liste


def enlever_doublon(liste):
    n = len(liste)
    new = []
    if n == 0:
        return []
    if n == 1:
        new.append(liste[0])
        return new
    else:
        for i in range(0, n - 1):
            if liste[i] != liste[i + 1]:
                new.append(liste[i])
        new.append(liste[n - 1])
    return new


def ellipse_final(liste):
    liste_0 = tracer_ellipse(liste)
    liste_1 = enlever_doublon(liste_0)
    return liste_1
