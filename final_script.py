#!/usr/bin/python3
import local_method
import global_method
import os
import pickle
import matplotlib.pyplot as plt

liste_image = pickle.load(open("picture/20.p", "rb"))


def change_array_in_list(tab):
    m = len(tab)
    n = len(tab[0])
    lst = []
    for i in range(0, m):
        for j in range(0, n):
            lst.append(tab[i, n - j])
    return lst


# uncomment if background is brighter than ring
liste_image = local_method.moins_tableau(liste_image)


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


# the gaussian blur is the blur that we have to apply to "smooth" the picture because the local method is using the second derivative so it's really sensitive to the noise.
# For a relatively smooth picture (18.p), a blur of 3 or 5 is good. For a picture with a bad quality you can have a blur of 11 or 13. It really depends on the "picture_with_features" you obtain.
local_method.gaussian_blur = 7

# the threshold is the filter for the features. The higher it is, the most selective it will be. If it's to high, the "picture_with_features" will be purple without any features.
# At the cuntrary, the lower it is, the less selective it will be. So if it is equal to zero, it won't be selective at all. The basic value really depends on the quality of the
# picture and the size of it but it's generally between 0.1 and 0.0001? You have to choose this value really carefully because you just want to select parts of the picture
# which are circles and not noise
local_method.threshold = 0.00017


# the confidence coefficient is a coefficient of the global method. In the global methodn we try to fit the shapes we see on the "picture_with_features" with real ellipse.
# this coefficient is a threshold so that if it is equal to 1 that means that the algorithm will only select ellipse which perfectly fits with the ellipse on the picture.
# if it's equal to 0.5, that means that only half of the ellipse has to fit with the shapes on the "picture_with_features".
# if the shapes of the electron rings are really close to ellipse you can have this coefficient equal to 0.8 but if the electron ring are far away from ellipse shapes,
# you'll have to take lowest value (it can be 0.2, it really depend on the quality of the picture and the shape of the electron rings)
global_method.confidence_coefficient = 0.98

# at each iteration, the algorithm is randomly choosing two edges of the "graph" (which means two part of circles) and try to see if those 2 edges are part of a single electron ring
# the algorithm is creating an ellipse which fit well with these two edges and if the fitting is good (the metric is give by the confidence coefficient), we will keep the ellipse
# if the fitting is bad, we won't keep the ellipse. The number of iterations is thus the number of times we are doing this operation
global_method.number_of_iterations = 500

# this  number is a threshold. After checking N possible ellipses (N is the number of iterations), the algorithm will return lots of ellipses which are good candidate to be real electron rings.
# if the algorithm has found more than the number of appearances the same ellipse, it will select this ellipse and it will return it as an ellipse of the picture
# if the algorithm has found more than the number of appearances the same ellipse, the algorithm will say that this fitting is not good so we won't select it
global_method.number_of_appearances = 2


# the global method is close to a RANSAC algorithm. We randomly choose some edges, we see if it gives an ellipse which fits well with the picture. If this ellipse has been created
# more than the number of appearances, we keep this ellipse, otherwise we don't take it into account.
local_method.local_main_procedure(liste_image)
global_method.global_main_procedure(liste_image)


# the result is a list : [[center_x,center_y,radius_x,radius_y,angle,confidence_coefficient,[liste of points]]]
