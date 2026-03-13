import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def flip_image(img):
    """Effectue un miroir horizontal."""
    return cv2.flip(img, 1)

def rotate_image(img, angle=45):
    """Pivote l'image avec un centre au milieu de l'image, en remplissant les bords vides avec du noir."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    # Calcule la matrice de rotation affine : centre, axe, échelle 1.0 (taille identique)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calcul des nouvelles dimensions (boîte englobante de l'image tournée)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Ajustement de la matrice pour pallier au recentrage original vs nouvelle boite
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    return cv2.warpAffine(img, rotation_matrix, (new_w, new_h))

def skew_image(img):
    """Applique une distorsion de perspective vers un parallélépipède."""
    h, w = img.shape[:2]
    # Points sources : les 4 coins de l'image (HG, HD, BG, BD en sens horaire/anti-horaire)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Points de destination : distorsion asymétrique
    offset = int(w * 0.2)
    pts2 = np.float32([[offset, 0], [w - offset, 0], [0, h], [w, h]])

    # Matrice de perspective pour mapper les points 1 aux points 2
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h))

def shear_image(img):
    """Applique un cisaillement (shear) horizontal (décalage du haut)."""
    h, w = img.shape[:2]
    
    # Matrice de transformation affine (shear horizontal)
    # [1, s, 0] # s = shear factor x
    # [0, 1, 0] # s = shear factor y (0 ici)
    shear_factor = 0.3
    matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    
    # Largeur aggrandie pour accueillir les pixels décalés par le cisaillement supérieur
    new_w = int(w + (h * shear_factor))
    return cv2.warpAffine(img, matrix, (new_w, h))

def crop_image(img):
    """Rogner ou zoomer une partie centrale (50% de la surface)."""
    h, w = img.shape[:2]
    target_h, target_w = int(h * 0.5), int(w * 0.5) # Crop au centre à 50%
    start_y, start_x = (h - target_h) // 2, (w - target_w) // 2
    return img[start_y:start_y+target_h, start_x:start_x+target_w]

def distort_image(img, power=0.5):
    """Effectue une distorsion en barillet (lens distortion)."""
    # Matrice caméra fictive au centre de l'image
    h, w = img.shape[:2]
    k_fx, k_fy = w * 0.5, h * 0.5 
    k_cx, k_cy = w / 2, h / 2
    camera_matrix = np.array([[k_fx, 0, k_cx], [0, k_fy, k_cy], [0, 0, 1]], dtype=np.float32)

    # Coefficients de distorsion radiale modifiés par power (ex: 0.5 pour gonfler, -0.5 pour creuser)
    dist_coeffs = np.array([power, power, 0, 0, 0], dtype=np.float32)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    return cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

def augment_image(image_path):
    """
    Génère 6 transformations pour une image donnée et les enregistre
    dans le même dossier avec les suffixes correspondants.
    """
    print(f"Applying data augmentation on {image_path}...")
    
    # Lecture OpenCV (attention BGR)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image at {image_path}")
        return
        
    orig_name, ext = os.path.splitext(image_path)
    
    # Dictionnaire des transformations et de l'instance d'image modifiée associée
    transformations = {
        "Flip": flip_image(img),
        "Rotate": rotate_image(img, 45),
        "Skew": skew_image(img),
        "Shear": shear_image(img),
        "Crop": crop_image(img),
        "Distortion": distort_image(img, -0.2)
    }

    # Création du plot avec Matplotlib
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle(f"Data Augmentation for {os.path.basename(image_path)}", fontsize=14)
    axes_flat = axes.flatten()

    for idx, (trans_name, modified_img) in enumerate(transformations.items()):
        # 1. Sauvegarde sur le disque avec suffixe (Leaffliction/dataset/Apple/apple_healthy/image (1)_Flip.JPG)
        out_filename = f"{orig_name}_{trans_name}{ext}"
        cv2.imwrite(out_filename, modified_img)
        print(f"Saved: {out_filename}")
        
        # 2. Ajout au Plot matplotlib
        # OpenCV étant BGR, plt.imshow fonctionne en RGB -> Conversion requise
        img_rgb = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)
        ax = axes_flat[idx]
        ax.imshow(img_rgb)
        ax.set_title(trans_name)
        ax.axis('off') # Désactive les numéros d'axes

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Applies data augmentation to a single leaf image."
    )
    parser.add_argument("image_path", help="Path to the source image")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: {args.image_path} is not a valid file.")
        return

    augment_image(args.image_path)


if __name__ == "__main__":
    main()
