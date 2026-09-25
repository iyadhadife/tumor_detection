from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch 

def evaluate_cnn_model(model, test_images, test_labels):
    """
    Évalue un modèle CNN PyTorch sur le jeu de test et calcule toutes les métriques clés.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for i in range(len(test_images)):
            img_tensor = test_images[i].unsqueeze(0).to(device)
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.cpu().item())
            true_labels.append(int(test_labels[i]))
            
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, zero_division=0)
    rec = recall_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds, zero_division=0)
    cm = confusion_matrix(true_labels, preds)
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Confusion Matrix": cm
    }
    return metrics, preds, true_labels