
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


#ago cluster
def dfs(image, x, y, visited, label):
    # Directions possibles pour se déplacer (haut, bas, gauche, droite)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Stack pour le DFS
    stack = [(x, y)]
    
    # Marquer ce pixel comme visité et lui attribuer un label
    visited[x, y] = True
    image[x, y] = label  # On marque le pixel avec son label
    
    while stack:
        cx, cy = stack.pop()
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:  # Vérifier si dans les limites
                if image[nx, ny] == 255 and not visited[nx, ny]:  # Si c'est un pixel blanc et non visité
                    visited[nx, ny] = True
                    image[nx, ny] = label  # Marquer ce pixel avec le même label
                    stack.append((nx, ny))

def find_connected_components(binary_image):
    # Créer une copie de l'image pour marquer les labels
    labeled_image = binary_image.copy()
    visited = np.zeros_like(binary_image, dtype=bool)  # Matrice pour marquer les pixels visités
    label = 1  # Le label à attribuer aux objets
    
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 255 and not visited[i, j]:
                # Si le pixel est blanc et n'a pas été visité, faire un DFS
                dfs(labeled_image, i, j, visited, label)
                label += 1  # Incrémenter le label pour le prochain objet trouvé

    return labeled_image

def calculate_center_of_mass(image, label):
    # Extraire les coordonnées des pixels correspondant à un label spécifique
    coords = np.column_stack(np.where(image == label))
    
    # Calculer le centre de masse (moyenne des coordonnées)
    center_of_mass = np.mean(coords, axis=0)
    
    return center_of_mass

def get_largest_cluster(image):
    # Compter les occurrences de chaque label
    unique, counts = np.unique(image, return_counts=True)
    
    # Ignorer le label 0 (fond) et trouver le label du plus gros cluster
    largest_label = unique[1:][np.argmax(counts[1:])]
    
    return largest_label


# Charger le fichier TIFF multi-pages
tiff_path = "/Users/constantindeumier/Documents/10sec_files3.tif"
img = Image.open(tiff_path)

# Liste pour stocker toutes les pages
images = []
positions = []
masses=[]

# Parcourir toutes les pages du TIFF
for i in range(img.n_frames):  
    img.seek(i)  # Aller à la page i
    img_array = np.array(img)  # Convertir en numpy
    images.append(img_array)

for image in images:
    # Trouver les pixels blancs (positions des objets)
    coords = np.column_stack(np.where(image >125))
  
    # Vérifier qu'il y a des pixels blancs
    if len(coords) == 0:
        print(f"Aucun objet trouvé dans {image}")
        continue
  
    # Appliquer DBSCAN pour regrouper les pixels et filtrer le bruit
    labeled_image = find_connected_components(image)
    # Identifier le plus grand cluster
    largest_label = get_largest_cluster(labeled_image)

    # Calculer le centre de masse du plus grand cluster
    center_of_mass = calculate_center_of_mass(labeled_image, largest_label)
    masses.append(center_of_mass[1])


    plt.imshow(labeled_image, cmap='nipy_spectral')
    plt.title("Composantes connexes")
    plt.show()

plt.plot(range(len(masses[1::2])),masses[1::2] , marker='o', linestyle='-', color='b')
plt.xlabel("Temps (ou index d'image)")
plt.ylabel("Position en y du centre de masse")
plt.title("Evolution de la position en y du centre de masse au fil du temps")
plt.grid(True)
plt.show()
