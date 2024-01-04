#%% md
# #  TP 6.5 Optimisation basée sur des tuners
# 
# ## Optimisation basée sur Keras tuner . 
# Le but est de ne pas teste toutes les solutions possibles (grid solution) (GridSearch). 
# - for param1 :
#     -  for parm2:
#         - for parm3
#             - ...
# 
# Le tuner propose des méthodes qui permettent d'accélérer la recherche de la meilleure solution. 
# Il propose 4 tuners
# - RandomSearch Tuner
# - GridSearch Tuner
# - BayesianOptimization Tuner
# - Hyperband Tuner
# - Sklearn Tuner
# 
# Il faut aller sur le site de keras (https://keras.io/api/keras_tuner/tuners/), pour comprendre ce que fait chacun de ces tuners (vous pourrez aussi le trouver sur tensorflow (https://www.tensorflow.org/tutorials/keras/keras_tuner).
# 
# il faut installer, keras-tuner.
# 
#%%
# pip install keras-tuner
# pip install tensorflow==2.3.0
# pip install tensorflow-gpu==2.3.0
# install keras 
# pip install keras==2.4.3
# pip install keras-tuner
# install pandas
# pip install pandas
# install sklearn
# pip install sklearn
#%%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import keras_tuner as kt  ## le keras tuner 
import pandas as pd
from sklearn.model_selection import train_test_split 
#%% md
# ### Préparation des données
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
#%% md
# #### Split des données
#%%
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test  
#%%
# Préparation ds données
# le -1 du load_data(-1) veut dire on prend toutes les lignes
data=load_data(-1)
# sélectionner les variables
X,y = select_variables(data)
X_train, X_test, y_train, y_test = split_data(X,y)
print("X_train.shape", X_train.shape, "X_test.shape", X_test.shape)
data.head()
#%% md
# 
#%%
## On peut utiliser une simple normalisation (x-mu)/ecart type)
def normaliser(X_train, X_test):
    mean = X_train.mean()
    std  = X_train.std()
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    return X_train, X_test 
#%%
# Préparation ds données
# le -1 du load_data(-1) veut dire on prend toutes les lignes 
data=load_data(-1)
# sélectionner les variables
X,y = select_variables(data)
X_train, X_test, y_train, y_test = split_data(X,y)
X_train, X_test = normaliser(X_train, X_test)
print("X_train.shape", X_train.shape, "X_test.shape", X_test.shape)
#%% md
# #### Définition du modèle.
# Je vous propose deux options, j'ai une préférence pour la deuxième option car on peut modifier le nombre de couches
#%% md
# #### Option 1 - les hyperparamètres à l'extérieur du modèle 
#%%
def create_model(neurons, lr, activations, hp_optimizers,optimizers_dict ): # type des données et leur significations en entrée, neurones : nommbres de neuronnes int , lr : learning rate float, activations : fonction d'activation str, hp_optimizers : optimizers str, optimizers_dict : dictionnaire des optimizers
    m = X_train.shape[1] # nombre de colonnes de X_train car c'est la taille de la couche d'entrée
    model = tf.keras.Sequential () # modèle séquentiel ou initialisation de l'objet modèle
    model.add(tf.keras.layers.Input(m,name="InputLayer")) # couche d'entrée
    model.add(tf.keras.layers.Dense(neurons, activation=activations)) # couche cachée
    model.add(tf.keras.layers.Dense(neurons, activation=activations)) # couche cachée
    model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid',name='output')) # couche de sortie
    # en gros on a construit un modèle avec 2 couches cachées de 16 neurones
    model.compile(optimizer=optimizers_dict[hp_optimizers], 
                  loss="BinaryCrossentropy", 
                  metrics=["accuracy"])
    return model

## Définir les différents paramètres à tester 
def build_model_opt1(hp):
    neurons = hp.Int("units", min_value=16, max_value=300, step=16)
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2)
    #p_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    activations=hp.Choice('activation',values=['tanh' ], default='tanh')
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-1)
    
    optimizers_dict = {
        "Adam":    tf.keras.optimizers.legacy.Adamax(learning_rate=learning_rate),
        "Adamax":  tf.keras.optimizers.legacy.Adamax(learning_rate=learning_rate),
        "SGD":     tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate),
        "Adagrad": tf.keras.optimizers.legacy.Adagrad(learning_rate=learning_rate)
        }

    hp_optimizers = hp.Choice('optimizer', values=["Adam","Adamax", "SGD", "Adagrad"])
    
    model = create_model(
        neurons=neurons, lr=lr, activations=activations, hp_optimizers=hp_optimizers, optimizers_dict=optimizers_dict 
    )
    return model
    
#%% md
# ### Option 2: les hyperparamètres sont définis dans le modèle
#%%
def build_model_opt2(hp): # hp : hyperparamètres, type de hp : kt.HyperParameters
    model = tf.keras.Sequential() # modèle séquentiel ou initialisation de l'objet modèle
    # Tune the number of layers.
    m = X_train.shape[1] # colonnes de x_train
    
    model = tf.keras.Sequential () # modèle séquentiel ou initialisation de l'objet modèle
    # couche d'entrée
    model.add(tf.keras.layers.Input(m,name="InputLayer")) # couche d'entrée
    # les;couches cachées
    for i in range(hp.Int("num_layers", 1, 5)): # permet de choisir un nombre de couches entre entre 1 et i=5
        model.add(
            tf.keras.layers.Dense( 
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=16, max_value=200, step=16),# permet de choisir le nombre de neurones entre 16 et 200
                activation=hp.Choice("activation", ["relu", "tanh"]),# permet de choisir la fonction d'activation entre relu et tanh
                )
        )
        # la couche de sortie 
    model.add(tf.keras.layers.Dense(1, activation="sigmoid")) # sigmoid car c'est un problème de classification binaire, sinon une regression on aurait mis linear, multi-class classification on aurait mis softmax
    
    # Liste hyperparameètres à optimiser   
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-1) #
    optimizers_dict = {
        "Adam":    tf.keras.optimizers.legacy.Adamax(learning_rate=learning_rate),
        "Adamax":  tf.keras.optimizers.legacy.Adamax(learning_rate=learning_rate),
        "SGD":     tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate),
        "Adagrad": tf.keras.optimizers.legacy.Adagrad(learning_rate=learning_rate)
        }

    hp_optimizers = hp.Choice('optimizer', values=["Adam","Adamax", "SGD", "Adagrad"])
    
    
    model.compile(
        optimizer=optimizers_dict[hp_optimizers],
        loss="BinaryCrossentropy",
        metrics=["accuracy"]
    )
    return model


build_model_opt2(kt.HyperParameters())

#%% md
# #### Choix du tuner ainsi que ses paramètres. 
# Conseil : visiter le site pour visualiser les différents paramètres du tuner (https://keras.io/api/keras_tuner/tuners/base_tuner/#tuner-class)
#%%
#tuner : Gridsearch, RandomSearch, BayesianOptimization, Hyperband
tuner = kt.BayesianOptimization(
    build_model_opt2,
    objective='val_accuracy',
    max_trials=16,
    overwrite=True,
    directory="\\./Tunes",
    project_name="tuning_BN",)


#%%

tuner = kt.Hyperband(
    build_model_opt2,
    objective='val_accuracy',
    max_epochs=16,
    factor=3,
    directory='\\./tuner_data',
    project_name='tuning_hyperband')

#%% md
# #### Visualiser les différents paramètres à tester
#%%
tuner.search_space_summary()
#%% md
# #### Lancer la méthode search du tuner avec ses paramètres pour rechercher les best paramètres
# Avant de lancer la méthode search on peut aussi lui demander de stopper la recherche si les résultats ne s'améliorent pas, ceci grace à (f.keras.callbacks.EarlyStopping) (https://keras.io/api/callbacks/early_stopping/)
#%%
## cette méthode est utile elle permet de stopper la recherche de solutions 
## quand l'erreur (ou la précision, ou ..), variable monitor= la loss, ne s'améliore pas 
## au bout de patience=5 epochs

early_stoping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,
    restore_best_weights=False)
#%%
tuner.search(X_train, y_train, epochs=32, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stoping])

#%%
tuner.results_summary()
#%% md
# #### Récupérer (get) les meilleurs paramètres, le meilleur modèle, ....
# (https://keras.io/api/keras_tuner/tuners/#the-base-tuner-class)
#%%
# le meilleur modèle est stocké en position [0] du get_best_model
best_model = tuner.get_best_models(num_models=1)[0]

# Les best paramètres
best_hps=tuner.get_best_hyperparameters()[0]

print("best #layers : ",best_hps.get('num_layers'))
print("best learning_rate : ",best_hps.get('learning_rate'))
print("best activation : ",best_hps.get('activation'))
print("best optimizer : ",best_hps.get('optimizer'))
best_model.summary()
#%% md
# #### Comment utiliser le meilleur modèle
# - Le meilleur modèle "best_model = tuner.get_best_models()" vient avec le mdèle de neurones déjà entrainé, les paramètres du modèle( W et les b) sont déjà appris. C'est ce que l'on nomme un "checkpoint". Ce modèle est à utiliser directement dans la phase d'évaluation(prédiction)
# - Sinon, le best_hps=tuner.get_best_hyperparameters()[0], lui récupère les meilleurs paramètres. Vous pourrez repartir de ces paramètres pour entrainer le modèle. (solution préconisée)
# 
#%%
## Sélectionner les meileurs hyperparamètres du modèle
best_model = tuner.hypermodel.build(best_hps)

# Réentrainer le modèle avec ls nouveaux hyperparamètres
history = best_model.fit(X_train, y_train, epochs=50, validation_data = (X_test, y_test), 
                         batch_size=32, 
                         verbose=False
                         callbacks=[early_stoping])


#%%
# AH quelle est la meilleure epoch ?? 
# Réupérer la best epoch
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#%%
history.history
#%%
## On peut aussi sélectionner les meileurs hyperparamètres
best_model = tuner.hypermodel.build(best_hps)

# on réentraine le modèle avec ls nouveaux hyperparamètres
best_model.fit(X_train, y_train, epochs=best_epoch, validation_data = (X_test, y_test), 
                         batch_size=32)
#%%
# Utilisation du best model
score = best_model.evaluate(X_test, y_test)
#%%
print("la loss",score[0], "l'accracy", score[1])
#%%
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(15,8))
plt.grid(True)
plt.show()
#%%
y_prob = best_model.predict(X_test)
y_classes = y_prob.argmax(axis=-1)

#%%
confusion_matrix = tf.math.confusion_matrix(y_test, y_classes)
#%%
import seaborn as sb    
class_names=[0,1]
# ax = plt.figure(figsize=(8, 6))
fig = sb.heatmap(confusion_matrix,  cmap='Greens')  

# labels, title and ticks
fig.set_xlabel('Predicted labels')
fig.set_ylabel('True labels')
fig.set_title('Confusion Matrix')
fig.xaxis.set_ticklabels(class_names) 
fig.yaxis.set_ticklabels(class_names)
fig.figure.set_size_inches(5, 5)


plt.show()