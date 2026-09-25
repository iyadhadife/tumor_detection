from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# Tentative d'import de XGBoost (gère gracieusement si non installé)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def get_classical_models(random_state=42):
    """
    Renvoie un dictionnaire contenant un large éventail de modèles de Machine Learning classique,
    incluant XGBoost si disponible.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "MLP Classifier": MLPClassifier(max_iter=500, random_state=random_state)
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        
    return models

class SimpleCNN(nn.Module):
    """
    Réseau de neurones convolutif (CNN) simple et efficace pour la classification d'images IRM (32x32).
    """
    def __init__(self, input_channels=3, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class DeepCNN(nn.Module):
    """
    Architecture CNN avancée, plus profonde (3 blocs convolutifs) avec Batch Normalization et Dropout.
    """
    def __init__(self, input_channels=3, num_classes=2):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Taille : 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Taille : 8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Taille : 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Module d'attention sur les canaux ('Quoi regarder ?')"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Module d'attention spatiale ('Où regarder ?' - cible la tumeur)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)

class CBAMBlock(nn.Module):
    """Bloc CBAM combinant l'attention de canal et spatiale"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class AttentionCNN(nn.Module):
    """
    CNN Avancé enrichi de blocs d'attention CBAM après chaque bloc convolutif,
    optimisé pour cibler précisément les zones tumorales sur les IRM.
    """
    def __init__(self, input_channels=3, num_classes=2):
        super(AttentionCNN, self).__init__()
        
        # Bloc 1 + Attention
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAMBlock(64),
            nn.MaxPool2d(2, 2) # 16x16
        )
        
        # Bloc 2 + Attention
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAMBlock(128),
            nn.MaxPool2d(2, 2) # 8x8
        )
        
        # Bloc 3 + Attention
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAMBlock(256),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        )
        
        # Classifieur final
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

import os
import torch
import torch.nn as nn
import torchvision.models as models

class TransferCNN(nn.Module):
    """
    Modèle de Transfer Learning basé sur DenseNet-121 ou ResNet-50,
    chargeant les poids pré-entraînés depuis un dossier local (ex: ../models).
    """
    def __init__(self, model_name='densenet121', num_classes=2, freeze_base=False, local_models_dir='../models'):
        super(TransferCNN, self).__init__()
        
        if model_name == 'densenet121':
            # Initialisation sans poids par défaut
            self.base_model = models.densenet121(weights=None)
            weight_path = os.path.join(local_models_dir, 'densenet121-a639ec97.pth')
            
            if os.path.exists(weight_path):
                print(f" Chargement des poids locaux DenseNet-121 depuis : {weight_path}")
                state_dict = torch.load(weight_path, map_location='cpu')
                # Filtrage de la couche de classification (1000 classes ImageNet) pour éviter le conflit de taille
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
                self.base_model.load_state_dict(state_dict, strict=False)
            else:
                print(f"⚠️ Attention : Fichier non trouvé dans {weight_path}, initialisation sans poids pré-entraînés.")
                
            if freeze_base:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                    
            # Remplacement par votre classifieur binaire (2 classes)
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
            
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(weights=None)
            weight_path = os.path.join(local_models_dir, 'resnet50-11ad3fa6.pth')
            
            if os.path.exists(weight_path):
                print(f" Chargement des poids locaux ResNet-50 depuis : {weight_path}")
                state_dict = torch.load(weight_path, map_location='cpu')
                # Filtrage de la couche fc (1000 classes ImageNet)
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
                self.base_model.load_state_dict(state_dict, strict=False)
            else:
                print(f"⚠️ Attention : Fichier non trouvé dans {weight_path}, initialisation sans poids pré-entraînés.")
                
            if freeze_base:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                    
            # Remplacement par votre classifieur binaire (2 classes)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError("Modèles supportés : 'densenet121', 'resnet50'")

    def forward(self, x):
        return self.base_model(x)

class ViTModel(nn.Module):
    """
    Vision Transformer (ViT-B/16) avec chargement local des poids pré-entraînés depuis ../models.
    Nécessite des images redimensionnées à 224x224 pixels.
    """
    def __init__(self, num_classes=2, freeze_base=False, local_models_dir='../models'):
        super(ViTModel, self).__init__()
        
        # Initialisation du modèle sans téléchargement automatique
        self.base_model = models.vit_b_16(weights=None)
        weight_path = os.path.join(local_models_dir, 'vit_b_16-c867db91.pth')
        
        if os.path.exists(weight_path):
            print(f" Chargement des poids locaux ViT-B/16 depuis : {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            # Filtrage de la tête de classification d'origine (1000 classes ImageNet)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('heads')}
            self.base_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ Attention : Fichier non trouvé dans {weight_path}, initialisation sans poids pré-entraînés.")
            
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # Remplacement par votre classifieur binaire (2 classes)
        in_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)