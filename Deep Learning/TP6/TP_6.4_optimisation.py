#%% md
#  # TP 6: Apprentissage profond(Deep learning)
#  ## TP 6.4  : Optimisation 
#  
# - Régularisation :  Dropout
# - Normalisation 
#     - Inputs
#     - les autres couches (batch Normalisation: avant ou après l'activation)
# - Optimiseurs: 
#     - RMSProp, Adam, SGD, 
# - Hyperparameter tuning : 
#     - learning_rate
#     - \#couches
#     - \#neurones par couche
#     - taille du mini batch 
# - ...
# # Exercice
# Trouver le meilleur modèle pour les données  de "smoking"
# Le programme doit tester les différentes configurations (et hyperparamètres).
# 
# Visiter le site keras ou tensorflow pour vérifier la maniène d'utiliser ces différents paramètres
# 
# PS : comparer aussi avec les modèes classiques : (KNN, Randomforest, ...) 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns

#%%
# Lecture du dataset n permet de limiter le nombre de lignes à lire
# Pour faciliter les tests

def load_data(n):
    data = pd.read_csv('../../data/train.csv')
    return data[0:n]

def select_variables(data):
    data.dropna(axis=0, inplace=True)
    y = data['smoking'] # récupérer la colonne survived et la mettre dans y
    # récuperer le reste des données dans X utiliser la fonction titanic.drop ???, ??? )
    X = data.drop('smoking', axis=1)
    return X,y

print("ok")
#%% md
# ### Définition du modèle 
# avec prise d'un nombre de couches et nombre de neurones variables 
#%%
def build_model(nb_layers, nb_units): # créer une fonction qui prend en paramètre le nombre de couches et le nombre de neurones par couche
    #utiliser Keras pour créer un modèle de type séquentiel
    # pour créer un model de A a Z il faut :  Sequential()
    # pour ajouter une couche : model.add(....)
    # implémentation : 
    input_shape = (X_train.shape[1],)
    model = keras.Sequential()
    model.add(keras.layers.Dense(nb_units, activation='relu', input_shape=input_shape))
    for i in range(nb_layers-1):
        model.add(keras.layers.Dense(nb_units, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model
#%%
#Compiler le modèle
#Optimiserer: SGD, AdamW, adadelta, ...
def compiler(model,optimizer,loss,metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

#%% md
# ### 4. Entrainement du modèle (Model training)
#%%
# Entrainement du modele
def train(model, X_train, y_train, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return history
    
    
    
    
    
#%% md
# ### 5. Evaluation des performances du modèle 
# 
#%%
def evaluer(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    return score

#print('Test loss     :', score[0])
#print('Test accuracy :', score[1])

#%%
def visualiser_confusion(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Get class labels
    y_classes = np.argmax(y_pred, axis=-1)

    cm = confusion_matrix(y_test, y_classes)
    #disp= ConfusionMatrixDisplay(confusion_matrix=cm)

    sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
    plt.show()

#%% md
# ### Main program
#%%
# Préparation ds données
# le -1 du load_data(-1) veut dire on prend toutes les lignes 
data=load_data(-1)
# sélectionner les variables
X,y = select_variables(data)


def split_data(X, y):
    # séparer les données en données d'entrainement et données de test
    # utiliser la fonction train_test_split de sklearn
    # X_train, X_test, y_train, y_test = train_test_split(....)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def normaliser(X_train, X_test):
    # normaliser les données d'entrainement et de test
    # utiliser la fonction RobustScaler de sklearn
    # X_train = RobustScaler().fit_transform(....)
    # X_test = RobustScaler().fit_transform(....)
    X_train = RobustScaler().fit_transform(X_train)
    X_test = RobustScaler().fit_transform(X_test)
    return X_train, X_test


X_train, X_test, y_train, y_test = split_data(X,y)
X_train, X_test = normaliser(X_train, X_test)

#%%


#%% md
# #### Afficher les paramètres du meilleur modèle
#%%
# Afficher les paramètres du meilleur modèle
best_param_model = {}
best_param_model["#layers"] = 2
best_param_model["#units"] = 10
best_param_model["optimizer"] = "adam"
best_param_model["loss"] = "binary_crossentropy"
best_param_model["metrics"] = ["accuracy"]
best_param_model["epochs"] = 100
print(best_param_model)

#%%
# Manière itérative de tester les différents paramètres et récupérer les best params

# 1. Tester les différents paramètres
def test_param(param, param_values): # en entrée param représente le nom du paramètre à tester et param_values représente les valeurs à tester
    best_score = 0 # on récupère le Math.max(best_score, score[1])
    best_param = None # on récupère le paramètre du best_score
    for param_value in param_values: # pour chaque valeur de param_values (chaque couche)
        best_param_model[param] = param_value # on met à jour le paramètre du modèle
        model = build_model(best_param_model["#layers"], best_param_model["#units"]) # on construit le modèle
        model = compiler(model, best_param_model["optimizer"], best_param_model["loss"], best_param_model["metrics"]) #compiler le modèle
        # on entraine le modèle et on l'attribue à history (historique des scores)
        history = train(model, X_train, y_train, best_param_model["epochs"], 32)
        # on évalue le modèle et on l'attribue à score
        score = evaluer(model, X_test, y_test)
        # on récupère le meilleur score et le meilleur paramètre
        if score[1] > best_score:
            best_score = score[1]
            best_param = param_value
    return best_param # une fois qu'on a fini d'itérer sur toutes les valeurs de param_values, on retourne le meilleur paramètre
#%%

#%%
best_param_model["#layers"]
#%% md
# #### Play with the best model
#%%
## sur les données de Text (X_test, y_test)
model = build_model(best_param_model["#layers"], best_param_model["#units"])
model = compiler(model, best_param_model["optimizer"], best_param_model["loss"], best_param_model["metrics"])
score = evaluer(model, X_test, y_test)
print('Test loss     :', score[0])
print('Test accuracy :', score[1])
#%%
print('Test loss     :', score[0])
print('Test accuracy :', score[1])

#Test loss     : 0.6390269994735718
#Test accuracy : 0.6051615476608276

# il faut faire mieux que ça !!

#%%
# changer les paramètres : 
optimizer = "adam"
loss = "binary_crossentropy"
metrics = ["accuracy"]
epochs = 100
batch_size = 32
#%%
# Entrainement du modèle
model = build_model(2, 10)
model = compiler(model, optimizer, loss, metrics)
history = train(model, X_train, y_train, epochs, batch_size)
#%%
# Evaluation du modèle
score = evaluer(model, X_test, y_test)
print('Test loss     :', score[0])
print('Test accuracy :', score[1])

#%% md
# Tester test_param sur les différents paramètres de train, donc il faut utiliser les data de train
#%%
train = load_data(-1)
X_train, y_train = select_variables(train)
X_train, X_test, y_train, y_test = split_data(X_train, y_train)
X_train, X_test = normaliser(X_train, X_test)
#%%
# Tester test_param sur les différents paramètres de train, donc il faut utiliser les data de train
# 1. Tester les différents paramètres
param = "#layers", "#units", "optimizer", "loss", "metrics", "epochs", "batch_size"
param_values = [2, 10, "adam", "binary_crossentropy", ["accuracy"], 100, 32]
best_param = test_param(param, param_values)
print(best_param)
#%%

#%%

X_train, X_test, y_train, y_test = split_data(X,y)
X_train, X_test = normaliser(X_train, X_test)

# Boucle de Test d'Hyperparamètres
nb_layers_options = [2, 3, 4]
nb_units_options = [10, 20, 30]
learning_rate_options = [0.01, 0.001, 0.0001]
batch_size_options = [16, 32, 64]

best_score = 0
best_params = {}

for nb_layers in nb_layers_options:
    for nb_units in nb_units_options:
        for lr in learning_rate_options:
            for batch_size in batch_size_options:
                model = build_model(nb_layers, nb_units)
                optimizer = keras.optimizers.Adam(learning_rate=lr)
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=0)
                score = model.evaluate(X_test, y_test, verbose=0)

                if score[1] > best_score:
                    best_score = score[1]
                    best_params = {
                        'layers': nb_layers,
                        'units': nb_units,
                        'learning_rate': lr,
                        'batch_size': batch_size
                    }

# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres:", best_params)

# Construire et Entraîner le Modèle avec les Meilleurs Hyperparamètres
model = build_model(best_params['layers'], best_params['units'])
optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=best_params['batch_size'], verbose=0)

# Effectuer une Prédiction
# Ici, remplacez `new_data` par vos nouvelles données d'entrée
new_data = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
new_data_normalized = RobustScaler().fit_transform(new_data.reshape(1, -1)) # Reshape si nécessaire
prediction = model.predict(new_data_normalized)
print("Prédiction (probabilité de fumer) :", prediction[0])

#%%
