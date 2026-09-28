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
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def get_classical_models(random_state=42):
    """
        Returns a dictionary containing a wide range of classical Machine Learning models,
        including XGBoost if available.
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
    Simple and efficient convolutional neural network (CNN) for MRI image classification (32x32).
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
    Advanced deeper CNN architecture (3 convolutional blocks) with Batch Normalization and Dropout.
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
            nn.MaxPool2d(2, 2),  # Size: 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Size: 8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # Size: 4x4
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
    """Channel attention module ('What to look at?')"""
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
    """Spatial attention module ('Where to look?') - targets the tumor"""
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
    """CBAM block combining channel and spatial attention"""
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
    Advanced CNN enhanced with CBAM attention blocks after each convolutional block,
    optimized to precisely target tumor regions in MRI scans.
    """
    def __init__(self, input_channels=3, num_classes=2):
        super(AttentionCNN, self).__init__()
        
        # Block 1 + Attention
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAMBlock(64),
            nn.MaxPool2d(2, 2)  # 16x16
        )
        
        # Block 2 + Attention
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAMBlock(128),
            nn.MaxPool2d(2, 2)  # 8x8
        )
        
        # Block 3 + Attention
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAMBlock(256),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )
        
        # Final classifier
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
    Transfer learning model based on DenseNet-121 or ResNet-50,
    loading pre-trained weights from a local folder (e.g., ../models).
    """
    def __init__(self, model_name='densenet121', num_classes=2, freeze_base=False, local_models_dir='../models'):
        super(TransferCNN, self).__init__()
        
        if model_name == 'densenet121':
            # Initialization without default weights
            self.base_model = models.densenet121(weights=None)
            weight_path = os.path.join(local_models_dir, 'densenet121-a639ec97.pth')
            
            if os.path.exists(weight_path):
                print(f" Loading local DenseNet-121 weights from: {weight_path}")
                state_dict = torch.load(weight_path, map_location='cpu')
                # Filtering the classification layer (1000 classes ImageNet) to avoid size mismatch
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
                self.base_model.load_state_dict(state_dict, strict=False)
            else:
                print(f"⚠️ Warning: File not found in {weight_path}, initializing without pre-trained weights.")
                
            if freeze_base:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                    
            # Replace with your binary classifier (2 classes)
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
            
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(weights=None)
            weight_path = os.path.join(local_models_dir, 'resnet50-11ad3fa6.pth')
            
            if os.path.exists(weight_path):
                print(f" Loading local ResNet-50 weights from: {weight_path}")
                state_dict = torch.load(weight_path, map_location='cpu')
                # Filtering the original fc layer (1000 classes ImageNet)
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
                self.base_model.load_state_dict(state_dict, strict=False)
            else:
                print(f"Warning: File not found in {weight_path}, initializing without pre-trained weights.")
                
            if freeze_base:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                    
            # Replace with your binary classifier (2 classes)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError("Supported models: 'densenet121', 'resnet50'")

    def forward(self, x):
        return self.base_model(x)

class ViTModel(nn.Module):
    """
    Vision Transformer (ViT-B/16) with local loading of pre-trained weights from ../models.
    Requires images resized to 224x224 pixels.
    """
    def __init__(self, num_classes=2, freeze_base=False, local_models_dir='../models'):
        super(ViTModel, self).__init__()
        
        # Initialize the model without automatic download
        self.base_model = models.vit_b_16(weights=None)
        weight_path = os.path.join(local_models_dir, 'vit_b_16-c867db91.pth')
        
        if os.path.exists(weight_path):
            print(f" Loading local ViT-B/16 weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            # Filtering the original classification head (1000 classes ImageNet)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('heads')}
            self.base_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: File not found in {weight_path}, initializing without pre-trained weights.")
            
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # Replace with your binary classifier (2 classes)
        in_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)