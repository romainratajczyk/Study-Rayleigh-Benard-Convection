

from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import correlate
from scipy.signal import correlate2d

from scipy.ndimage import uniform_filter
import scipy.ndimage

corr = []

tiff_path = "/Users/constantindeumier/Desktop/10sec_files2.tif"
img = Image.open(tiff_path)

# Liste pour stocker toutes les pages
images = []
imageslissées = []
positions = []
masses=[]
DX = []
DY = []
# Parcourir toutes les pages du TIFF
for i in range(img.n_frames):  
    img.seek(i)  # Aller à la page i
    img_array = np.array(img) - np.mean(img) # Convertir en numpy
    images.append(img_array)


    
def compute_displacement_grid(img1, img2, step=50, window_size=100):
    """
    Quadrille l'image et calcule les déplacements locaux par corrélation.
    
    img1, img2 : images de référence et décalée (matrices NumPy)
    step : distance entre les points du quadrillage
    window_size : taille de la fenêtre autour de chaque point
    """
    h, w = img1.shape
    half_win = window_size // 2
    
    # Listes pour stocker les coordonnées et déplacements
    X, Y, U, V = [], [], [], []

    for y in range(half_win, h - half_win, step):
        for x in range(half_win, w - half_win, step):
            # Extraire la fenêtre autour du point (x, y)
            window1x = img1[y, x-half_win:x+half_win]
            window2x = img2[y, x-half_win:x+half_win]
            window1y = img1[y-half_win:y+half_win, x]
            window2y = img2[y-half_win:y+half_win, x]
            # Centrer les intensités (réduit l'effet de luminosité)
            window1x = window1x - np.mean(window1x)
            window2x = window2x - np.mean(window2x)
            window1y = window1y - np.mean(window1y)
            window2y = window2y - np.mean(window2y)
            # Calcul de la corrélation
            correlationx = np.abs(correlate(window1x, window2x, mode="same"))
            correlationy = np.abs(correlate(window1y, window2y, mode="same"))
            # Trouver le pic de corrélation
           # Trouver les indices des maxima dans la corrélation
            dy = np.argmax(correlationy)
            dx = np.argmax(correlationx)
            
            # Ajuster les déplacements (soustraction de la moitié de la taille de la corrélation)
            dy -= correlationy.shape[0] // 2
            dx -= correlationx.shape[0] // 2


            # Stocker les résultats
            if not np.isnan(dx) and not np.isnan(dy):
               X.append(x)
               Y.append(y)
               U.append(dx)
               V.append(dy)


    return np.array(X).astype(float), np.array(Y).astype(float), np.array(U).astype(float), np.array(V).astype(float)



def compute_displacement_grid2(img1, img2, step=50, window_size=100):
    """
    Quadrille l'image et calcule les déplacements locaux par corrélation 2D.

    img1, img2 : images de référence et décalée (matrices NumPy)
    step : distance entre les points du quadrillage
    window_size : taille de la fenêtre autour de chaque point
    """
    h, w = img1.shape
    half_win = window_size // 2
    
    # Listes pour stocker les coordonnées et déplacements
    X, Y, U, V = [], [], [], []

    for y in range(half_win, h - half_win, step):
        for x in range(half_win, w - half_win, step):
            # Extraire la fenêtre autour du point (x, y)
            window1 = img1[y-half_win:y+half_win+1, x-half_win:x+half_win+1].astype(float)
            window2 = img2[y-half_win:y+half_win+1, x-half_win:x+half_win+1].astype(float)

            if window1.shape != (window_size, window_size) or window2.shape != (window_size, window_size):
                print('oui')
                continue  # Évite les problèmes aux bords

            # Centrer les intensités pour minimiser l'effet de luminosité
            window1 = window1 - np.mean(window1)
            window2 = window2 - np.mean(window2)

            # Corrélation 2D
            correlation = np.abs(correlate2d(window1, window2, mode="same"), dtype=float)
            correlation /= np.max(correlation) 
            corr.append(correlation)
            

            # Trouver le maximum de corrélation
            dy, dx = np.unravel_index(np.argmax(correlation), correlation.shape)

            # Ajuster les déplacements (référence au centre de la fenêtre)
            dy -= correlation.shape[1] // 2
            dx -= correlation.shape[0] // 2

            # Stocker les résultats
            if not np.isnan(dx) and not np.isnan(dy):
                X.append(x)
                Y.append(y)
                U.append(dx)
                V.append(dy)
                DX.append(dx)
                DY.append(dy)
                

    return np.array(X, dtype=float), np.array(Y, dtype=float), np.array(U, dtype=float), np.array(V, dtype=float)


# Création de la figure
fig, ax = plt.subplots(figsize=images[0].shape)
im = ax.imshow(images[0], cmap="gray")

# Affichage interactif
plt.ion()  

'''
for i in range((len(images)//2)-1):
    XX, YY, UU, VV = compute_displacement_grid(images[2*i], images[2*i+1], step=40, window_size=35)
    plt.figure(figsize=(6,6))
    im.set_array(images[2*i])
    plt.draw()
    plt.pause(0.1)
    plt.quiver(XX, YY, 3*UU, 3*VV, color="red", angles="xy", scale_units="xy", scale=1, width=0.001)  # Affichage du champ de vecteurs
    plt.title("Champ de vecteurs du déplacement")
'''
'''for i in range((len(images)//2)-1):
    XX, YY, UU, VV = compute_displacement_grid(images[2*i], images[2*i+1], step=40, window_size=35)
'''

# Initialisation du champ de vecteurs
XX, YY, UU, VV = compute_displacement_grid2(images[0], images[1], step=10, window_size=15)
quiver = ax.quiver(XX, YY, UU, VV, color="red", angles="xy", scale_units="xy", scale=1, width=0.0002)


def average_vectors(XX, YY, UU, VV, kernel_size=15):
    """
    Moyenne les vecteurs (UU, VV) dans un voisinage carré de taille kernel_size autour de chaque flèche.
    
    XX, YY : Coordonnées des flèches
    UU, VV : Composantes des vecteurs
    kernel_size : Taille du voisinage (doit être impair)
    
    Retourne : U_avg, V_avg (vecteurs moyennés)
    """
    # Convertir en flottants et en matrice 2D
    grid_shape = (len(np.unique(YY)), len(np.unique(XX)))  # Dimensions de la grille
    U_matrix = UU.astype(float).reshape(grid_shape)
    V_matrix = VV.astype(float).reshape(grid_shape)

    # Appliquer un filtre de moyenne sur une fenêtre kernel_size x kernel_size
    U_avg = uniform_filter(U_matrix, size=kernel_size, mode='constant', cval=0)
    V_avg = uniform_filter(V_matrix, size=kernel_size, mode='constant', cval=0)

    # Aplatir pour retrouver le format d'origine
    return U_avg.ravel(), V_avg.ravel()
# Fonction de mise à jour de l'animation
def update(frame):
    im.set_array(images[2*frame])  # Mettre à jour l'image affichée
    XX, YY, UU, VV = compute_displacement_grid2(images[2*frame], images[2*frame+1], step=20, window_size=31)
        
    # Mettre à jour les vecteurs
    # Supprime l'ancien champ de vecteurs
    UUU, VVV = average_vectors(XX, YY, UU.astype(float), VV.astype(float), kernel_size=3)
    quiver = ax.quiver(XX, YY, UUU, VVV, color="red", scale_units="xy", scale=0.05, width=0.001)
    im.set_array(images[2*frame+1])
    return [im, quiver]  # Retourne les objets mis à jour``



# Créer l'animation
ani = animation.FuncAnimation(fig, update, frames=img.n_frames//2, interval=100, blit=True)

# Afficher l'animation
plt.show()

print(corr[59])
print(np.unravel_index(np.argmax(corr[59]), corr[59].shape))
print(corr[0][dx][dy])
'''
# Création de la figure
# Activer l'affichage interactif (nécessaire pour certains environnements)
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(images[0], cmap="gray")

# Fonction de mise à jour pour l'animation
def update(frame):
    im.set_array(images[frame])
    return [im]
'''
'''# Créer et afficher l'animation
ani = animation.FuncAnimation(fig, update, frames=len(images), interval=100)
for img in images:
    im.set_array(img)
    plt.draw()
    plt.pause(0.1)  # Pause pour afficher l'image





'''






