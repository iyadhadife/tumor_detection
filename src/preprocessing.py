import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def create_dataset_dict(dataset, names):
    for name in names:
        dataset[name] = []
    return dataset

def add_to_dataset(dataset, value, name):
    dataset[name].append(value)
    return dataset

def color_names(names, nom):
    for i in range(256):
        names.append(f'{nom}{i}')
    return names

def add_color_to_dataset(dataset, hist, color):
    for i in range(256):
        dataset[color + f'{i}'].append(hist[i][0])
    return dataset

def compute_mean(tab):
    return sum(tab) / len(tab) if len(tab) > 0 else 0

def extract_image_components(path):
    areas = []
    perimeters = []
    moments = []
    dataset = {}
    
    names = [
        'moyenne_des_laplaciens',
        "nombre_d_aires",
        "moyenne_des_perimetres",
        "nombre_de_moments",
        "moyenne_des_aires",
        "max_des_areas",
        "max_des_perimeters",
        "min_des_areas",
        "min_des_perimeters"
    ]
    names = color_names(names, 'couleurs_bleues')
    names = color_names(names, 'couleurs_vertes')
    names = color_names(names, 'couleurs_rouges')
    names = color_names(names, 'couleurs_grises')
    
    dataset = create_dataset_dict(dataset, names)
    
    if not os.path.exists(path):
        print(f"Directory not found: {path}")
        return dataset

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        tsr_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        tsr_img_color = cv2.imread(file_path, cv2.IMREAD_COLOR)
        
        if tsr_img is None or tsr_img_color is None:
            continue
            
        # Thresholding for binary image
        _, thresh = cv2.threshold(tsr_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Histograms
        histgray = cv2.calcHist([tsr_img], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([tsr_img_color], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([tsr_img_color], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([tsr_img_color], [2], None, [256], [0, 256])
        
        # Texture feature using Laplacian
        laplacian = cv2.Laplacian(tsr_img, cv2.CV_64F)
        laplacian_mean = float(np.mean(np.abs(laplacian)))
        
        M = {}
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            areas.append(area)
            perimeters.append(perimeter)
            M = cv2.moments(contour)
            moments.append(M)
            
        number_of_areas = len(areas)
        number_of_moments = len(M) if M else 0
        mean_perimeters = compute_mean(perimeters)
        mean_areas = compute_mean(areas)
        max_area = max(areas) if areas else 0
        max_perimeter = max(perimeters) if perimeters else 0
        min_area = min(areas) if areas else 0
        min_perimeter = min(perimeters) if perimeters else 0

        dataset = add_to_dataset(dataset, laplacian_mean, 'moyenne_des_laplaciens')
        dataset = add_color_to_dataset(dataset, hist_b, 'couleurs_bleues')
        dataset = add_color_to_dataset(dataset, hist_g, 'couleurs_vertes')
        dataset = add_color_to_dataset(dataset, hist_r, 'couleurs_rouges')
        dataset = add_color_to_dataset(dataset, histgray, 'couleurs_grises')
        dataset = add_to_dataset(dataset, number_of_areas, "nombre_d_aires")
        dataset = add_to_dataset(dataset, number_of_moments, "nombre_de_moments")
        dataset = add_to_dataset(dataset, mean_perimeters, "moyenne_des_perimetres")
        dataset = add_to_dataset(dataset, mean_areas, "moyenne_des_aires")
        dataset = add_to_dataset(dataset, max_area, "max_des_areas")
        dataset = add_to_dataset(dataset, max_perimeter, "max_des_perimeters")
        dataset = add_to_dataset(dataset, min_area, "min_des_areas")
        dataset = add_to_dataset(dataset, min_perimeter, "min_des_perimeters")
        
        areas = []
        perimeters = []
        moments = []
        
    return dataset

def create_nsame_values(val, nb):
    return [val] * nb

def preprocess_tabular_data(df, target_col='labels', n_components=30, apply_smote=True):
    """
    Cleans, scales, balances (via SMOTE), and reduces dimensions (via PCA) of tabular features.
    
    Parameters:
    - df (pd.DataFrame): The raw tabular dataframe containing features and labels.
    - target_col (str): Name of the target column.
    - n_components (int): Number of principal components to keep for PCA.
    - apply_smote (bool): Whether to apply SMOTE for balancing classes.
    
    Returns:
    - X_pca (np.ndarray): Preprocessed and reduced feature matrix.
    - y_balanced (np.ndarray): Balanced target labels.
    - pca (PCA): Fitted PCA object.
    - scaler (StandardScaler): Fitted scaler object.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert string labels ('yes'/'no') to binary integers (1/0)
    if y.dtype == object or y.isin(['yes', 'no']).any():
        y = y.apply(lambda val: 1 if str(val).lower() in ['yes', '1', 'true'] else 0)

    # Handle missing values by filling with 0 (since missing areas/colors imply a count/value of 0)
    X = X.fillna(0)
    
    # 1. Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Data Balancing using SMOTE (Synthetic Minority Over-sampling Technique)
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    else:
        X_balanced, y_balanced = X_scaled, y.values
        
    # 3. Dimensionality Reduction using PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_balanced)
    
    return X_pca, y_balanced, pca, scaler

def augment_tabular_data(X, y, noise_factor=0.01):
    """
    Applique une data augmentation sur les caractéristiques tabulaires 
    en ajoutant un léger bruit gaussien pour enrichir la diversité du dataset d'entraînement.
    """
    noise = np.random.normal(0, noise_factor, X.shape)
    X_augmented = X + noise
    X_combined = np.vstack((X, X_augmented))
    y_combined = np.hstack((y, y))
    return X_combined, y_combined

def preprocess_tabular_data(df, target_col='labels', n_components=30, apply_smote=True, apply_augmentation=True):
    """
    Nettoie, normalise, équilibre (via SMOTE), augmente (via bruit gaussien) 
    et réduit les dimensions (via PCA à 30 composants pour ~98% de variance) des données tabulaires.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if y.dtype == object or y.isin(['yes', 'no']).any():
        y = y.apply(lambda val: 1 if str(val).lower() in ['yes', '1', 'true'] else 0)

    X = X.fillna(0)
    
    # 1. Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Équilibrage des classes (SMOTE)
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    else:
        X_balanced, y_balanced = X_scaled, y.values
        
    # 3. Data Augmentation tabulaire (Bruit gaussien optionnel)
    if apply_augmentation:
        X_balanced, y_balanced = augment_tabular_data(X_balanced, y_balanced, noise_factor=0.01)

    # 4. Réduction de dimension par PCA (30 composants pour ~98% de variance)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_balanced)
    
    return X_pca, y_balanced, pca, scaler