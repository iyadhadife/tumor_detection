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

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import copy as std_copy

def train_and_evaluate_models(models_dict, X_train, X_test, y_train, y_test):
    performance_data = []
    trained_models = {}
    predictions = {}
    confusion_matrices = {}
    
    for name, model in models_dict.items():
        print(f"--- Training {name} ---")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
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

def train_cnn_model(model, train_images, train_labels, epochs=30, lr=0.0001):
    """
    Entraîne le modèle CNN avec PyTorch sur le dataset d'images augmentées.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer)
    
    model.train()
    dataloader = pd.DataFrame({'images': train_images, 'labels': train_labels})
    
    history = {'epoch': [], 'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(dataloader.shape[0]):
            optimizer.zero_grad()
            img_tensor = dataloader['images'].iloc[i].unsqueeze(0)
            label_tensor = dataloader['labels'].iloc[i].unsqueeze(0)
            
            outputs = model(img_tensor)
            loss = criterion(outputs, label_tensor)
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += label_tensor.size(0)
            correct += (predicted == label_tensor).sum().item()
            
        scheduler.step()
        epoch_acc = 100 * correct / total if total > 0 else 0
        avg_loss = running_loss / dataloader.shape[0]
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['accuracy'].append(epoch_acc)
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    return model, history

def train_cnn_model(model, train_images, train_labels, epochs=30, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer)
    
    model.train()
    dataloader = pd.DataFrame({'images': train_images, 'labels': train_labels})
    
    history = {'epoch': [], 'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(dataloader.shape[0]):
            optimizer.zero_grad()
            img_tensor = dataloader['images'].iloc[i].unsqueeze(0)
            label_tensor = dataloader['labels'].iloc[i].unsqueeze(0)
            
            outputs = model(img_tensor)
            loss = criterion(outputs, label_tensor)
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += label_tensor.size(0)
            correct += (predicted == label_tensor).sum().item()
            
        scheduler.step()
        epoch_acc = 100 * correct / total if total > 0 else 0
        avg_loss = running_loss / dataloader.shape[0]
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['accuracy'].append(epoch_acc)
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    return model, history

def train_advanced_cnn_with_early_stopping(
    model, 
    train_images, train_labels, 
    val_images, val_labels, 
    optimizer_type='AdamW', 
    lr=0.001, 
    epochs=40, 
    batch_size=32, 
    patience=6
):
    """
    Entraîne n'importe quel modèle CNN (SimpleCNN ou DeepCNN) avec :
    - Optimiseurs paramétrables (AdamW, Adam, SGD).
    - Scheduler de Learning Rate (ReduceLROnPlateau).
    - Early Stopping basé sur l'accuracy de validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_dataset = TensorDataset(torch.stack(train_images), torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(torch.stack(val_images), torch.tensor(val_labels, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    best_weights = std_copy.deepcopy(model.state_dict())  # Utilisation de std_copy
    patience_counter = 0
    
    history = {'train_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        history['train_loss'].append(epoch_train_loss)
        history['val_accuracy'].append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = std_copy.deepcopy(model.state_dict())  # Utilisation de std_copy
            patience_counter = 0
            print(f"   ✨ Meilleure accuracy validation : {best_val_acc:.2f}% (Poids sauvegardés)")
        else:
            patience_counter += 1
            print(f"   ⏳ Stagnation ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"🛑 Early stopping déclenché à l'époque {epoch+1} !")
                break
                
    model.load_state_dict(best_weights)
    return model, history, best_val_acc