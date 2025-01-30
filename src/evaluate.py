import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def confusion_matrix_plot(X, y_true, y_pred, split='Testing'):
    """Plots a confusion matrix for the given data.

    Args:
        X (np.array): Input features.
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        split (str, optional): Data split name. Defaults to 'Testing'.
    """
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"{split} Accuracy: {accuracy:.3f}")

    # Create the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # The confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.title(f'Accuracy = {accuracy:.3f}')
    plt.suptitle(f'Confusion Matrix - {split} Data')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return plt 