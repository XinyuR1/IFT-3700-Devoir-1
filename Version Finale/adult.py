# DEVOIR 1 - CODES
# Muxue Guo et Ronnie Liu
# Fichier adult.py
# Fichier qui teste la performance des cinq algorithmes sur ADULT
# avec la notion de dissimilarité originale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import algorithms as al

from sklearn.model_selection import train_test_split
#########################################################
#######################  ADULT  #########################
#########################################################

print('ADULT')
print('ADULT: DISSIMILARITÉ ORIGINALE')

# NOTE: Pour MNIST, on compare la précision avec la distance euclidienne. Étant
# donné que ADULT contient des données catégorielles, nous avons décidé de comparer
# notre performance des cinq algorithmes avec la précision obtenur dans la démo 2
# qui est environ de 81,0% de précision par kNN.

adult = pd.read_csv("adult.csv")
# On enlève les rangées avec les données manquantes. On travaille avec les données
# avec aucun point d'interrogation.
adult = adult[(adult != '?').all(axis = 1)]

# Échantillonnage des données ADULT

# On va prendre les 2400 premières données du fichier adult.csv (excluant les données ayant un champ '?')
# On va ensuite séparer en 2000 pour apprentissage (qui sera séparée encore en 1600-400 pour entraînement et validation)
# et en 400 pour les données de test.

n_total = 2400
adult = adult.head(n_total)
y = adult['income']
adult = adult.drop(['income'], axis = 1)

x_trainval, x_test, y_trainval, y_test = train_test_split(adult, y, test_size = (1/6), stratify=y, random_state=0)
x_train, x_valid, y_train, y_valid = train_test_split(x_trainval, y_trainval, test_size = (1/5), random_state=0)

x_trainval = x_trainval.values
x_train = x_train.values
x_valid = x_valid.values
x_test = x_test.values

# Remplacer les labels par 0 et 1 pour facilier les manipulations
y_trainval = y_trainval.replace({'<=50K': 0, '>50K': 1}).values
y_train = y_train.replace({'<=50K': 0, '>50K': 1}).values
y_valid = y_valid.replace({'<=50K': 0, '>50K': 1}).values
y_test = y_test.replace({'<=50K': 0, '>50K': 1}).values

# Vérification des dimensions des features
print(x_trainval.shape)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)


# TEST DE MAJORITÉ
print(f'Nombre de personnes avec salaire <= 50K: {np.count_nonzero(y_test == 0)}')

y_train_majority = np.zeros(len(y_train))
y_test_majority = np.zeros(len(y_test))

acc_majority = al.calculate_accuracy(y_train, y_test, y_train_majority, y_test_majority)

print(f'\nTEST DE MAJORITÉ:')
print(f'Précision sur les données de test: {acc_majority[1]}%')



# MATRICE DE DISSIMILARITÉ ADULT
print('\nMATRICE DE DISSIMILARITÉ ADULT (Apprentissage)')
adult_diss_trainval = al.generate_matrix(al.dissimilarity_adult, x_trainval)

print('\nMATRICE DE DISSIMILARITÉ ADULT (Entraînement)')
adult_diss_train = al.generate_matrix(al.dissimilarity_adult, x_train)

print('\nMATRICE DE DISSIMILARITÉ ADULT (Validation)')
adult_diss_valid = al.generate_matrix(al.dissimilarity_adult, x_valid, x_train)

print('\nMATRICE DE DISSIMILARITÉ ADULT (Test)')
adult_diss_test = al.generate_matrix(al.dissimilarity_adult, x_test, x_trainval)



# Algorithme 1: KMédoïdes
print('\nALGORITHME 1: KMédoïdes')
adult_kmedoids = al.predict_kmedoids([0, 1], adult_diss_trainval, adult_diss_test)
acc_kmedoids = al.calculate_accuracy(y_trainval, y_test, adult_kmedoids[0], adult_kmedoids[1])

print(f'Précision sur les données d\'apprentissage: {acc_kmedoids[0]}%')
print(f'Précision sur les données de test: {acc_kmedoids[1]}%')



# Algorithme 2: Partition Binaire
print('\nALGORITHME 2: Partition Binaire')
adult_agglo = al.predict_agglo(10, adult_diss_trainval, adult_diss_test)
acc_agglo = al.calculate_accuracy(y_trainval, y_test, adult_agglo[0], adult_agglo[1])

print(f'Précision sur les données d\'apprentissage: {acc_agglo[0]}%')
print(f'Précision sur les données de test: {acc_agglo[1]}%')



# Algorithme 3: KNN
print('\nALGORITHME 3: kNN')

# Tester pour différents valeurs d'hyperparamètres
voisins = np.array([1, 2, 5, 10, 20, 50, 100])
acc_train = np.zeros(len(voisins))
acc_valid = np.zeros(len(voisins))

for i in range(len(voisins)):
    adult_knn = al.predict_knn(voisins[i], adult_diss_train, adult_diss_valid, y_train, True)
    acc_knn = al.calculate_accuracy(y_train, y_valid, adult_knn[0], adult_knn[1])

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
adult_knn = al.predict_knn(20, adult_diss_trainval, adult_diss_test, y_trainval, True)
acc_knn = al.calculate_accuracy(y_trainval, y_test, adult_knn[0], adult_knn[1])

print(f'Hyperparamètre choisi: 20 voisins')
print(f'Précision sur les données d\'apprentissage: {acc_knn[0]}%')
print(f'Précision sur les données de test: {acc_knn[1]}%')



# Algorithme 4: PCoA
print('\nALGORITHME 4: PCoA')

# Tester pour différents valeurs d'hyperparamètres
components = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
acc_train = np.zeros(len(components))
acc_valid = np.zeros(len(components))

for i in range(len(components)):
    adult_pcoa = al.predict_pcoa(components[i], adult_diss_train, adult_diss_valid, 20, y_train)
    acc_pcoa = al.calculate_accuracy(y_train, y_valid, adult_pcoa[0], adult_pcoa[1])

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
adult_pcoa = al.predict_pcoa(10, adult_diss_trainval, adult_diss_test, 20, y_trainval)
acc_pcoa = al.calculate_accuracy(y_trainval, y_test, adult_pcoa[0], adult_pcoa[1])

print(f'Hyperparamètre choisi: 10 composantes principales et 20 voisins kNN')
print(f'Précision sur les données d\'apprentissage: {acc_pcoa[0]}%')
print(f'Précision sur les données de test: {acc_pcoa[1]}%')



# Algorithme 5: Isomap
print('\nALGORITHME 5 Isomap')

# Tester pour différents valeurs d'hyperparamètres
components = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
acc_train = np.zeros(len(components))
acc_valid = np.zeros(len(components))

for i in range(len(components)):
    adult_isomap = al.predict_isomap(components[i], adult_diss_train, adult_diss_valid, 20, y_train, 4)
    acc_isomap = al.calculate_accuracy(y_train, y_valid, adult_isomap[0], adult_isomap[1])

    acc_train[i] = acc_isomap[0]
    acc_valid[i] = acc_isomap[1]

# Graphique des précisions d'entraînement et de validation
plt.plot(components, acc_train, linestyle='--', marker='o', color='b', label="Précision d'entraînement")
plt.plot(components, acc_valid, linestyle='--', marker='o', color='r', label="Précision de validation")
plt.legend(frameon=True, loc='lower center', ncol=2)
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Accuracy')
plt.title('Précision de classification de Isomap')
plt.grid(True)
plt.show()

# Prédiction sur la matrice de dissimilarité de test-trainval avec l'hyperparamètre optimal
adult_isomap = al.predict_isomap(9, adult_diss_trainval, adult_diss_test, 20, y_trainval, 4)
acc_isomap = al.calculate_accuracy(y_trainval, y_test, adult_isomap[0], adult_isomap[1])

print(f'Hyperparamètre choisi: 9 composantes et 20 voisins kNN')
print(f'Précision sur les données d\'apprentissage: {acc_isomap[0]}%')
print(f'Précision sur les données de test: {acc_isomap[1]}%')