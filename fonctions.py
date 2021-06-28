from keras.engine.sequential import Sequential
import numpy as np
import shutil
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.models import load_model

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------


def list_folder(directory):
    """
    Fonction permettant de lister le nom des dossiers (est utile juste une fois pour la fonction "clean_lite")
    """
    nom_dossier = os.listdir(directory + "/test")
    return nom_dossier

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------


def clean_lite(nombre, directory):
    """
    Fonction permettant d'alléger les données dans un soucis d'optimisation
    """
    CATEGORIES = list_folder("data/cifar-100")

    LITE_CATEGORIES = np.random.choice(CATEGORIES, nombre)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(directory + "/train"):
        os.makedirs(directory + "/train")

    if not os.path.exists(directory + "/test"):
        os.makedirs(directory + "/test")

        for lc in LITE_CATEGORIES:
            if not os.path.exists(directory + f"/train/{lc}"):
                os.makedirs(directory + f"/train/{lc}")
            if not os.path.exists(directory + f"/test/{lc}"):
                os.makedirs(directory + f"/test/{lc}")

                file_list_train = os.listdir(f"data/cifar-100/train/{lc}/")
                file_list_test = os.listdir(f"data/cifar-100/test/{lc}/")

                for file in file_list_train:
                    shutil.copy(f"data/cifar-100/train/{lc}/{file}", directory + f"/train/{lc}")
                for file in file_list_test:
                    shutil.copy(f"data/cifar-100/test/{lc}/{file}", directory + f"/test/{lc}")
    else:
        print("Tout les fichiers nécessaires sont déjà présents! --> 'data\cifar_lite'")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------


def display_activation(activations, col_size, row_size, act_index):
    """
    Fonction qui permet le plot des layers
    """
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(
        row_size*13.5, col_size*2.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='plasma')
            activation_index += 1
    plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------


def list_index(directory):
    """
    Fonction permettant de lister les index des catégories dans le directory
    """

    listdir = os.listdir(directory + "/test")
    list_index = []
    for obj in listdir:
        list_index.append(listdir.index(obj))
    return list_index


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_model(model, image, directory):
    """
    Fonction qui nous permet de tester notre modèle
    """

    listindex = list_index(directory)

    model = load_model(model)
    print("\nModèle chargé!")

    test_image = plt.imread(image)

    resized_image = resize(test_image, (32, 32, 3))
    plt.imshow(resized_image)
    predictions = model.predict(np.array([resized_image]))
    predictions
    x = predictions
    classification = list_folder(directory)

    plt.show()

    print("\nRoulement de tambour...\nPrédictions :\n")

    for i in range(len(listindex)):
        for j in range(len(listindex)):
            if x[0][listindex[i]] > x[0][listindex[j]]:
                temp = listindex[i]
                listindex[i] = listindex[j]
                listindex[j] = temp

    # Affiche les étiquettes triées dans l'ordre de la probabilité de la plus élevée à la plus faible
    # print(list_index)

    i=0
    for i in range(len(listindex)):
        print(classification[listindex[i]], ':', round(predictions[0][listindex[i]] * 100, 2), '%')
    print("\n")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
