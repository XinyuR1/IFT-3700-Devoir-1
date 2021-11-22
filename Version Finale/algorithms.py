# DEVOIR 1 - CODES
# Muxue Guo et Ronnie Liu
# Fichier algorithms.py
# Fichier qui stocke tous les algorithmes utilisés sur les deux datasets
# incluant les cinq algorithmes (supervisé et non-supervisé).
import numpy as np

from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier

# pip install scikit-image pour télécharger ce package
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks

#########################################################
###################   ALGORITHMES  ######################
#########################################################

# FONCTION DE CALCUL DU "ACCURACY"
# Fonction qui permet de calculer l'exactitude de l'algorithme (en %)
# Fonction de perte utilisée: erreur 0-1

# y1: y_train
# y2: y_valid ou y_test
# y_hat1: prédictions de y_train
# y_hat2: prédictions de y_test

# Retourne: un tuple contenant la précision sur les données d'entraînement
# ainsi que la précision sur les données de validation/de test
def calculate_accuracy(y1, y2, y_hat1, y_hat2):
    error_1 = 0.0
    error_2 = 0.0

    # Erreur d'entraînement
    for i in range(len(y1)):
        if y1[i] != y_hat1[i]:
            error_1 = error_1 + 1.0

    error_1 = (error_1 / len(y1)) * 100

    # Erreur de validation/de test
    for i in range(len(y2)):
        if y2[i] != y_hat2[i]:
            error_2 = error_2 + 1.0

    error_2 = (error_2 / len(y2)) * 100

    return (100 - error_1, 100 - error_2)


# MESURES DE SIMILARITÉ

# Distance euclidienne
# Source: TP4
def l2(x, y):
    return np.sum(abs(x - y) ** 2) ** (1 / 2)


# Similarité MNIST

# Alpha: Détermine le poids accordé à la détection de lignes versus la détection de cercles
# i.e. distance totale = alpha * distance_lignes + (1-alpha) * distance_cercles

# Nb_angles: Nombre d'angles testés pour la détection de lignes
# Un nombre trop grand d'angles peut détecter de fausses lignes, mais
# un nombre trop petit d'angles peut ne pas détecter des lignes importantes

# Tested_R: Rayons de cercles testés lors de la recherche de cercles
# Inclure des cercles trop petits peut nuire l'algorithme en négligeant les plus grands cercles
# moins complets que les petits cercles
def dissimilarity_mnist(x, y, alpha=0.5, nb_angles=180, tested_R=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
    # Réduire l'image à son squelette (segment d'épaisseur 1)
    img1 = skeletonize(x.reshape((28, 28)))
    img2 = skeletonize(y.reshape((28, 28)))

    # Détection de lignes, avec l'accumulateur des lignes stockés dans h1 et h2
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, nb_angles, endpoint=False)
    h1, theta1, d1 = hough_line(img1, theta=tested_angles)
    h2, theta2, d2 = hough_line(img2, theta=tested_angles)

    # Détection de cercles, avec l'accumulateur des cercles dans H1_c et H2_c
    H1_c = hough_circle(img1, tested_R)
    H2_c = hough_circle(img2, tested_R)

    # Normaliser les valeurs entre 0 et 255
    h1 *= int(255.0 / h1.max())
    h2 *= int(255.0 / h2.max())

    H1_c *= int(255.0 / H1_c.max())
    H2_c *= int(255.0 / H2_c.max())

    return alpha * l2(h1.astype(np.int64), h2.astype(np.int64)) + (1 - alpha) * l2(H1_c.astype(np.int64),
                                                                                   H2_c.astype(np.int64))


# Similarité ADULT

# Weight: ajuste les calculs de similarité pour chaque attribut en
# ajoutant ou enlevant plus de poids dépendant la situation.

# Alpha: calcul de similarité entre deux individus pour chaque attribut
# Pour un attribut catégorique: 0 ou 1
# Pour un attribut numérique: [0,1]

# Epsilon: Le threshold (seuil) pour déterminer la valeur d'alpha numérique
# Notez que lorsque epsilon = 0, cela indique que c'est un attribut catégorique.

def dissimilarity_adult(array_1, array_2):
    weight = np.array([1, 2, 0, 0, 1.5, 0.5, 2, 0.5, 1, 0.75, 0, 0, 1.5, 1])
    epsilon = np.array([10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 10, 0])
    dissimilarity = 0

    for i in range(weight.shape[0]):
        alpha = 0
        if epsilon[i] == 0:
            # Lorsque les deux catégories sont différents
            if array_1[i] != array_2[i]:
                alpha = 1
                dissimilarity = dissimilarity + weight[i] * alpha
        else:
            difference = abs(array_1[i] - array_2[i])
            # Lorsque la différence ne dépasse pas le seuil
            if difference < epsilon[i]:
                alpha = float(difference) / epsilon[i]
            else:
                alpha = 1
            dissimilarity = dissimilarity + weight[i] * alpha

    return dissimilarity



# Matrice de dissimilarité
def get_dissimilarity_matrix(dist, X, Y):
    diss_matrix = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            diss_matrix[i, j] = dist(X[i], Y[j])

        # Indique l'utilisateur l'itération à laquelle nous sommes rendus.
        if (i + 1) % 50 == 0:
            print(f'Itération {i + 1} terminée!')
        if (i + 1) == X.shape[0]:
            print(f'Fin du calcul de la matrice')

    return diss_matrix



# Génération de la matrice de dissimilarité
# x et y: les features d'entraînement/de validation/de test
def generate_matrix(dist, X, Y=None):
    Y = X if Y is None else Y
    diss_matrix = get_dissimilarity_matrix(dist, X, Y)

    # Imprimer la dimension de la matrice afin d'assurer la taille de cette dernière
    print(f'Dimension de la matrice: {diss_matrix.shape}')
    return diss_matrix


# 5 ALGORITHMES
# À noter que tous les algorithmes ont comme sortie un tuple où:
# 1er élément: les prédictions des données d'apprentissage
# 2e élément: les prédictions des données de test

# KMédoïdes
# Source: Exemple du code travail 1
def predict_kmedoids(initial_medoids, diss_train, diss_test):
    kmedoids_instance = kmedoids(diss_train, initial_medoids, data_type='distance_matrix')
    kmedoids_instance.process()

    # Prédictions
    kmedoids_train = kmedoids_instance.predict(diss_train)
    kmedoids_test = kmedoids_instance.predict(diss_test)

    return (kmedoids_train, kmedoids_test)


# Partition Binaire
# Clustering par partition binaire
# Source: Exemple du code travail 1
def agglomerative_clustering_predict(agglomerative_clustering, dissimilarity_matrix):
    average_dissimilarity = list()
    for i in range(agglomerative_clustering.n_clusters):
        ith_clusters_dissimilarity = dissimilarity_matrix[:, np.where(agglomerative_clustering.labels_ == i)[0]]
        average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
    return np.argmin(np.stack(average_dissimilarity), axis=0)

def predict_agglo(cluster, diss_train, diss_test):
    agglomerative_clustering = AgglomerativeClustering(n_clusters=cluster, affinity='precomputed', linkage='average')
    agglomerative_clustering.fit(diss_train)

    agglo_train = agglomerative_clustering_predict(agglomerative_clustering, diss_train)
    agglo_test = agglomerative_clustering_predict(agglomerative_clustering, diss_test)

    return (agglo_train, agglo_test)


# kNN
# train, test peuvent être soit x_train/x_test ou les matrices de dissimilarité
def predict_knn(neighbors, train, test, y_train, is_matrix):
    # is_matrix détermine si l'entrée est une matrice de dissimilarité ou non. Sinon, cela veut dire qu'on utilise
    # kNN pour prédire les labels des exemples modifiés par des méthodes non supervisées (PCoA ou Isomap)
    if is_matrix:
        knn = KNeighborsClassifier(n_neighbors=neighbors, metric='precomputed', algorithm='brute')
    else:
        knn = KNeighborsClassifier(n_neighbors=neighbors)

    knn.fit(train, y_train)

    # Prédictions
    knn_train = knn.predict(train)
    knn_test = knn.predict(test)

    return (knn_train, knn_test)

# NOTE: Puisque PCoA et Isomap sont des méthodes non supervisées, on mesure la précision de cette méthode
# en utilisant un classifieur kNN sur les données modifiées par PCoA et Isomap.

# PCoA
# Source: Exemple du code travail 1
def predict_pcoa(component, diss_train, diss_test, neighbors, y_train):
    pcoa = KernelPCA(n_components=component, kernel='precomputed')

    # Modification des valeurs des features
    pcoa_train = pcoa.fit_transform(-.5 * diss_train ** 2)
    pcoa_test = pcoa.transform(-.5 * diss_test ** 2)

    # Prédictions par kNN
    return predict_knn(neighbors, pcoa_train, pcoa_test, y_train, False)


# Isomap
# Source: Exemple du code travail 1
# neighbors: voisins pour isomap
# neighbors_nn: voisins pour knn afin de prédire le modèle avec isomap
def predict_isomap(component, diss_train, diss_test, neighbors_knn, y_train, neighbors):
    isomap = Isomap(n_components=component, n_neighbors=neighbors, metric='precomputed')
    isomap_train = isomap.fit_transform(diss_train)
    isomap_test = isomap.transform(diss_test)

    return predict_knn(neighbors_knn, isomap_train, isomap_test, y_train, False)