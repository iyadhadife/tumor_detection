import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate_models(models_dict, X_train, X_test, y_train, y_test):
    """
    Entraîne une liste de modèles et évalue leurs performances sur le jeu de test.
    
    Returns:
    - results_df (pd.DataFrame): Tableau récapitulatif des métriques trié par Accuracy.
    - trained_models (dict): Dictionnaire des modèles entraînés.
    - predictions (dict): Dictionnaire des prédictions.
    - confusion_matrices (dict): Dictionnaire des matrices de confusion.
    """
    performance_data = []
    trained_models = {}
    predictions = {}
    confusion_matrices = {}
    
    for name, model in models_dict.items():
        print(f"--- Entraînement de {name} ---")
        # Entraînement
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Prédiction
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calcul des métriques
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        
        performance_data.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
    results_df = pd.DataFrame(performance_data)
    results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    return results_df, trained_models, predictions, confusion_matrices