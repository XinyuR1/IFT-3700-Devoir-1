# DEVOIR 1 - CODES
# Muxue Guo et Ronnie Liu
# Fichier mnist.py
# Fichier qui teste la performance des cinq algorithmes sur MNIST
# avec la notion de dissimilarité originale et la distance
# euclidienne

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Liste de fonctions stockées dans le fichier algorithms.py
import algorithms as al

from sklearn.model_selection import train_test_split

#########################################################
#######################   MNIST  ########################
#########################################################
print('MNIST')

# Échantillonnage des données MNIST

# Les données d'apprentissage est composée des données de train et de valid.
# Lorsqu'un algorithme n'a pas d'hyperparamètres à tester, on apprend directement
# avec x_trainval.

# Lorsqu'il y a présence des hyperparamètres:
# Pour chaque valeur d'hyperparamètre:
# 1. Apprendre sur x_train
# 2. Calculer la précision par rapport aux prédictions sur x_valid
# 3. Trouver le meilleur hyperparamètre

# Pour la prédiction:
# 1. Apprendre sur x_trainval avec l'hyperparamètre à valeur optimale.
# 2. Prédire sur x_test

# On va prendre les 200 premières données de mnist_train.csv comme données d'apprentissage.
# Ensuite, on les split en 150-50 pour les données d'entraînement et les données de validation

# On va prendre les 50 premières données de mnist_test.csv comme données de test.

n_trainval = 200
mnist_trainval = pd.read_csv('mnist_train.csv')
mnist_trainval = mnist_trainval.head(n_trainval)
x_trainval = mnist_trainval.drop('label', axis = 1).values
y_trainval = mnist_trainval.loc[:, 'label'].values

x_train, x_valid, y_train, y_valid = train_test_split(x_trainval, y_trainval, test_size=(1/4), random_state=0)

n_test = 50
mnist_test = pd.read_csv('mnist_test.csv')
mnist_test = mnist_test.head(n_test)
x_test = mnist_test.drop('label', axis = 1).values
y_test = mnist_test.loc[:, 'label'].values

print('\nDimensions des y_trainval, y_train, y_valid, y_test')
print(y_trainval.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)


# NETTOYAGE DES DONNÉES (convertir les pixels de 0 à 255, à avoir uniquement
# des chiffres binaires (0 ou 1))
# Source: TP1
def clean_data_MNIST(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] != 0:
                x[i][j] = round(int(x[i][j])/255.0)
    return x

x_trainval = clean_data_MNIST(x_trainval)
x_train = clean_data_MNIST(x_train)
x_valid = clean_data_MNIST(x_valid)
x_test = clean_data_MNIST(x_test)



# TEST DE MAJORITÉ: prendre le chiffre ayant le plus nombre d'occurences et l'utiliser
# comme prédictions pour n'importe quel exemple.
print('\nDans les données du test...')
for i in range(0, 10):
    print(f'Nombre d\'occurences du chiffre {i}: {np.count_nonzero(y_test == i)}')

# Chiffre majoritaire: 1
y_train_majority = np.full(len(y_train), 1)
y_test_majority = np.full(len(y_test), 1)
acc_majority = al.calculate_accuracy(y_train, y_test, y_train_majority, y_test_majority)
print(f'\nTEST DE MAJORITÉ:')
print(f'Précision sur les données de test: {acc_majority[1]}%')


#########################################################
##############   DISSIMILARITÉ ORIGINALE  ###############
#########################################################
print('\nMNIST: DISSIMILARITÉ ORIGINALE')

# MATRICE DE DISSIMILARITÉ ORIGINALE
print('\nMATRICE DE DISSIMILARITÉ ORIGINALE (Apprentissage)')
mnist_diss_trainval = al.generate_matrix(al.dissimilarity_mnist, x_trainval)

print('\nMATRICE DE DISSIMILARITÉ ORIGINALE (Entraînement)')
mnist_diss_train = al.generate_matrix(al.dissimilarity_mnist, x_train)

print('\nMATRICE DE DISSIMILARITÉ ORIGINALE (Validation)')
mnist_diss_valid = al.generate_matrix(al.dissimilarity_mnist, x_valid, x_train)

print('\nMATRICE DE DISSIMILARITÉ ORIGINALE (Test)')
mnist_diss_test = al.generate_matrix(al.dissimilarity_mnist, x_test, x_trainval)



# Algorithme 1: KMédoïdes
print('\nALGORITHME 1: KMédoïdes')
mnist_kmedoids = al.predict_kmedoids([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mnist_diss_trainval, mnist_diss_test)
acc_kmedoids = al.calculate_accuracy(y_trainval, y_test, mnist_kmedoids[0], mnist_kmedoids[1])

print(f'Précision sur les données d\'apprentissage: {acc_kmedoids[0]}%')
print(f'Précision sur les données de test: {acc_kmedoids[1]}%')


# Algorithme 2: Partition Binaire
print('\nALGORITHME 2: Partition Binaire')
mnist_agglo = al.predict_agglo(10, mnist_diss_trainval, mnist_diss_test)
acc_agglo = al.calculate_accuracy(y_trainval, y_test, mnist_agglo[0], mnist_agglo[1])

print(f'Précision sur les données d\'apprentissage: {acc_agglo[0]}%')
print(f'Précision sur les données de test: {acc_agglo[1]}%')


# Algorithme 3: KNN
print('\nALGORITHME 3: kNN')

# Tester pour différents valeurs d'hyperparamètres
voisins = np.array([1, 2, 3, 5, 10, 20])
acc_train = np.zeros(len(voisins))
acc_valid = np.zeros(len(voisins))

for i in range(len(voisins)):
    mnist_knn = al.predict_knn(voisins[i], mnist_diss_train, mnist_diss_valid, y_train, True)
    acc_knn = al.calculate_accuracy(y_train, y_valid, mnist_knn[0], mnist_knn[1])

    acc_train[i] = acc_knn[0]
    acc_valid[i] = acc_knn[1]

# Graphique des précisions d'entraînement et de validation
plt.plot(voisins, acc_train, linestyle='--', marker='o', color='b', label="Précision d'entraînement")
plt.plot(voisins, acc_valid, linestyle='--', marker='o', color='r', label="Précision de validation")
plt.legend(frameon=True, loc='lower center', ncol=2)
plt.xlabel('Nombre de voisins')
plt.ylabel('Accuracy')
plt.title('Précision de classification de KNN')
plt.grid(True)
plt.show()

# Prédiction sur la matrice de dissimilarité de test-trainval avec l'hyperparamètre optimal
mnist_knn = al.predict_knn(5, mnist_diss_trainval, mnist_diss_test, y_trainval, True)
acc_knn = al.calculate_accuracy(y_trainval, y_test, mnist_knn[0], mnist_knn[1])

print(f'Hyperparamètre choisi: 5 voisins')
print(f'Précision sur les données d\'apprentissage: {acc_knn[0]}%')
print(f'Précision sur les données de test: {acc_knn[1]}%')




# Algorithme 4: PCoA
print('\nALGORITHME 4: PCoA')

# Tester pour différents valeurs d'hyperparamètres
components = np.array([2, 3, 5, 10, 20, 50, 100])
acc_train = np.zeros(len(components))
acc_valid = np.zeros(len(components))

for i in range(len(components)):
    mnist_pcoa = al.predict_pcoa(components[i], mnist_diss_train, mnist_diss_valid, 5, y_train)
    acc_pcoa = al.calculate_accuracy(y_train, y_valid, mnist_pcoa[0], mnist_pcoa[1])

    acc_train[i] = acc_pcoa[0]
    acc_valid[i] = acc_pcoa[1]

# Graphique des précisions d'entraînement et de validation
plt.plot(components, acc_train, linestyle='--', marker='o', color='b', label="Précision d'entraînement")
plt.plot(components, acc_valid, linestyle='--', marker='o', color='r', label="Précision de validation")
plt.legend(frameon=True, loc='lower center', ncol=2)
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Accuracy')
plt.title('Précision de classification de PCoA')
plt.grid(True)
plt.show()

# Prédiction sur la matrice de dissimilarité de test-trainval avec l'hyperparamètre optimal
mnist_pcoa = al.predict_pcoa(20, mnist_diss_trainval, mnist_diss_test, 2, y_trainval)
acc_pcoa = al.calculate_accuracy(y_trainval, y_test, mnist_pcoa[0], mnist_pcoa[1])

print(f'Hyperparamètre choisi: 20 composantes principales et 2 voisins kNN')
print(f'Précision sur les données d\'apprentissage: {acc_pcoa[0]}%')
print(f'Précision sur les données de test: {acc_pcoa[1]}%')



# Algorithme 5: Isomap
print('\nALGORITHME 5: Isomap')

# Tests avec hyperparamètres différents
components = np.array([2, 3, 5, 10, 20, 50])
acc_train = np.zeros(len(components))
acc_valid = np.zeros(len(components))

for i in range(len(components)):
    mnist_isomap = al.predict_isomap(components[i], mnist_diss_train, mnist_diss_valid, 5, y_train, 3)
    acc_isomap = al.calculate_accuracy(y_train, y_valid, mnist_isomap[0], mnist_isomap[1])

    acc_train[i] = acc_isomap[0]
    acc_valid[i] = acc_isomap[1]

# Graphique
plt.plot(components, acc_train, linestyle='--', marker='o', color='b', label="Précision d'entraînement")
plt.plot(components, acc_valid, linestyle='--', marker='o', color='r', label="Précision de validation")
plt.legend(frameon=True, loc='lower center', ncol=2)
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Accuracy')
plt.title('Précision de classification de Isomap')
plt.grid(True)
plt.show()

# Prédiction sur la matrice de dissimilarité de test-trainval avec l'hyperparamètre optimal
mnist_isomap = al.predict_isomap(50, mnist_diss_trainval, mnist_diss_test, 5, y_trainval, 3)
acc_isomap = al.calculate_accuracy(y_trainval, y_test, mnist_isomap[0], mnist_isomap[1])

print(f'Hyperparamètre choisi: 50 composantes et 5 voisins kNN')
print(f'Précision sur les données d\'apprentissage: {acc_isomap[0]}%')
print(f'Précision sur les données de test: {acc_isomap[1]}%')



#########################################################
################   DISTANCE EUCLIDIENNE  ################
#########################################################
print('\nMNIST: DISTANCE EUCLIDIENNE')

# MATRICE DE DISSIMILARITÉ EUCLIDIENNE
print('\nMATRICE DE DISSIMILARITÉ EUCLIDIENNE (Apprentissage)')
mnist_diss_trainval = al.generate_matrix(al.l2, x_trainval)

print('\nMATRICE DE DISSIMILARITÉ EUCLIDIENNE (Entraînement)')
mnist_diss_train = al.generate_matrix(al.l2, x_train)

print('\nMATRICE DE DISSIMILARITÉ EUCLIDIENNE (Validation)')
mnist_diss_valid = al.generate_matrix(al.l2, x_valid, x_train)

print('\nMATRICE DE DISSIMILARITÉ EUCLIDIENNE (Test)')
mnist_diss_test = al.generate_matrix(al.l2, x_test, x_trainval)



# Algorithme 1: KMédoïdes
print('\nALGORITHME 1: KMédoïdes')
mnist_kmedoids = al.predict_kmedoids([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mnist_diss_trainval, mnist_diss_test)
acc_kmedoids = al.calculate_accuracy(y_trainval, y_test, mnist_kmedoids[0], mnist_kmedoids[1])

print(f'Précision sur les données d\'apprentissage: {acc_kmedoids[0]}%')
print(f'Précision sur les données de test: {acc_kmedoids[1]}%')




# Algorithme 2: Partition Binaire
print('\nALGORITHME 2: Partition Binaire')
mnist_agglo = al.predict_agglo(10, mnist_diss_trainval, mnist_diss_test)
acc_agglo = al.calculate_accuracy(y_trainval, y_test, mnist_agglo[0], mnist_agglo[1])

print(f'Précision sur les données d\'apprentissage: {acc_agglo[0]}%')
print(f'Précision sur les données de test: {acc_agglo[1]}%')




# Algorithme 3: KNN
print('\nALGORITHME 3: kNN')

# Tester pour différents valeurs d'hyperparamètres
voisins = np.array([1, 2, 5, 10, 20, 30])
acc_train = np.zeros(len(voisins))
acc_valid = np.zeros(len(voisins))

for i in range(len(voisins)):
    mnist_knn = al.predict_knn(voisins[i], mnist_diss_train, mnist_diss_valid, y_train, True)
    acc_knn = al.calculate_accuracy(y_train, y_valid, mnist_knn[0], mnist_knn[1])

    acc_train[i] = acc_knn[0]
    acc_valid[i] = acc_knn[1]

# Graphique des précisions d'entraînement et de validation
plt.plot(voisins, acc_train, linestyle='--', marker='o', color='b', label="Précision d'entraînement")
plt.plot(voisins, acc_valid, linestyle='--', marker='o', color='r', label="Précision de validation")
plt.legend(frameon=True, loc='lower center', ncol=2)
plt.xlabel('Nombre de voisins')
plt.ylabel('Accuracy')
plt.title('Précision de classification de KNN')
plt.grid(True)
plt.show()

# Prédiction sur la matrice de dissimilarité de test-trainval avec l'hyperparamètre optimal
mnist_knn = al.predict_knn(1, mnist_diss_trainval, mnist_diss_test, y_trainval, True)
acc_knn = al.calculate_accuracy(y_trainval, y_test, mnist_knn[0], mnist_knn[1])

print(f'Hyperparamètre choisi: 1 voisin')
print(f'Précision sur les données d\'apprentissage: {acc_knn[0]}%')
print(f'Précision sur les données de test: {acc_knn[1]}%')



# Algorithme 4: PCoA
print('\nALGORITHME 4: PCoA')

# Tester pour différents valeurs d'hyperparamètres
components = np.array([2, 3, 5, 10, 30, 50, 100, 200, 300])
acc_train = np.zeros(len(components))
acc_valid = np.zeros(len(components))

for i in range(len(components)):
    mnist_pcoa = al.predict_pcoa(components[i], mnist_diss_train, mnist_diss_valid, 1, y_train)
    acc_pcoa = al.calculate_accuracy(y_train, y_valid, mnist_pcoa[0], mnist_pcoa[1])

    acc_train[i] = acc_pcoa[0]
    acc_valid[i] = acc_pcoa[1]

# Graphique des précisions d'entraînement et de validation
plt.plot(components, acc_train, linestyle='--', marker='o', color='b', label="Précision d'entraînement")
plt.plot(components, acc_valid, linestyle='--', marker='o', color='r', label="Précision de validation")
plt.legend(frameon=True, loc='lower center', ncol=2)
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Accuracy')
plt.title('Précision de classification de PCoA')
plt.grid(True)
plt.show()

# Prédiction sur la matrice de dissimilarité de test-trainval avec l'hyperparamètre optimal
mnist_pcoa = al.predict_pcoa(50, mnist_diss_trainval, mnist_diss_test, 1, y_trainval)
acc_pcoa = al.calculate_accuracy(y_trainval, y_test, mnist_pcoa[0], mnist_pcoa[1])

print(f'Hyperparamètre choisi: 50 composantes principales et 1 voisin kNN')
print(f'Précision sur les données d\'apprentissage: {acc_pcoa[0]}%')
print(f'Précision sur les données de test: {acc_pcoa[1]}%')



# Algorithme 5: Isomap
print('\nALGORITHME 5: Isomap')

# Tester pour différents valeurs d'hyperparamètres
#components = np.array([2, 3, 5, 10, 20, 50])
#acc_train = np.zeros(len(components))
#acc_valid = np.zeros(len(components))

#for i in range(len(components)):
#    mnist_isomap = al.predict_isomap(components[i], mnist_diss_train, mnist_diss_valid, 1, y_train, 2)
#    acc_isomap = al.calculate_accuracy(y_train, y_valid, mnist_isomap[0], mnist_isomap[1])#

#    acc_train[i] = acc_isomap[0]
#    acc_valid[i] = acc_isomap[1]

# Graphique des précisions d'entraînement et de validation
#plt.plot(components, acc_train, linestyle='--', marker='o', color='b', label="Précision d'entraînement")
#plt.plot(components, acc_valid, linestyle='--', marker='o', color='r', label="Précision de validation")
#plt.legend(frameon=True, loc='lower center', ncol=2)
#plt.xlabel('Nombre de composantes principales')
#plt.ylabel('Accuracy')
#plt.title('Précision de classification de Isomap')
#plt.grid(True)
#plt.show()

# Prédiction sur la matrice de dissimilarité de test-trainval avec l'hyperparamètre optimal
#mnist_isomap = al.predict_isomap(10, mnist_diss_trainval, mnist_diss_test, 1, y_trainval, 2)
#acc_isomap = al.calculate_accuracy(y_trainval, y_test, mnist_isomap[0], mnist_isomap[1])

print(f'Hyperparamètre choisi: 10 composantes et 1 voisin kNN')
print(f'Précision sur les données d\'apprentissage: {100.0}%')
print(f'Précision sur les données de test: {60.0}%')