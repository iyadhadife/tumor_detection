import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def plot_model_comparison(results_df):
    """
    Affiche un graphique à barres comparant l'accuracy des différents modèles.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
    plt.title("Comparaison de l'Accuracy des Modèles de Machine Learning")
    plt.xlim(0, 1.05)
    plt.xlabel("Accuracy")
    plt.ylabel("Modèle")
    plt.tight_layout()
    plt.show()

def plot_metrics_heatmap(results_df):
    """
    Affiche une heatmap de toutes les métriques (Accuracy, Precision, Recall, F1-Score).
    """
    df_melt = results_df.set_index('Model')
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_melt, annot=True, cmap='Blues', fmt=".4f", linewidths=.5)
    plt.title("Heatmap comparative des métriques par modèle")
    plt.ylabel("Modèle")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(confusion_matrices_dict):
    """
    Affiche toutes les matrices de confusion sous forme de grille pour analyse d'erreurs.
    """
    n_models = len(confusion_matrices_dict)
    cols = 3
    rows = (n_models // cols) + (1 if n_models % cols != 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()
    
    for i, (name, cm) in enumerate(confusion_matrices_dict.items()):
        ax = axes[i]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Tumor', 'Tumor'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(name)
        ax.grid(False)
        
    # Masquer les axes vides restants s'il y en a
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()