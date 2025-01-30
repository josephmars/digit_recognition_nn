import numpy as np
from data_preprocessing import load_and_preprocess_data, apply_pca
from train import initialize_model, train_model
from neural_network import NeuralNetwork
from evaluate import confusion_matrix_plot
import matplotlib.pyplot as plt

def main():
    # Set random seed for reproducibility
    np.random.seed(0)

    # Load and preprocess data
    filepath = "./data/optdigits-orig.windep"
    (X_train, X_test, X_val, y_train, y_test, y_val,
     y_train_one_hot, y_test_one_hot, y_val_one_hot) = load_and_preprocess_data(filepath)

    print(f"Data Shapes:\n"
          f"X_train: {X_train.shape}, X_test: {X_test.shape}, X_val: {X_val.shape},\n"
          f"y_train: {y_train.shape}, y_test: {y_test.shape}, y_val: {y_val.shape}")

    # Initialize and train the neural network
    input_size = X_train.shape[1]
    model = initialize_model(input_size=input_size, hidden_size=20, output_size=10)
    train_model(model, X_train, y_train_one_hot, X_val, y_val_one_hot, epochs=200, learning_rate=0.01)

    # Evaluation on Training Data
    y_true_train = y_train.flatten()
    y_pred_train = np.argmax(model.forward(X_train), axis=1)
    plot = confusion_matrix_plot(X_train, y_true_train, y_pred_train, split='Training')
    plot.savefig('./plots/confusion_matrix_training.png')

    # Evaluation on Validation Data
    y_true_val = y_val.flatten()
    y_pred_val = np.argmax(model.forward(X_val), axis=1)
    plot = confusion_matrix_plot(X_val, y_true_val, y_pred_val, split='Validation')
    plot.savefig('./plots/confusion_matrix_validation.png')

    # Evaluation on Testing Data
    y_true_test = y_test.flatten()
    y_pred_test = np.argmax(model.forward(X_test), axis=1)
    plot = confusion_matrix_plot(X_test, y_true_test, y_pred_test, split='Testing')
    plot.savefig('./plots/confusion_matrix_testing.png')

    # Bonus: Dimensionality Reduction with PCA
    X_train_pca, X_test_pca, X_val_pca = apply_pca(X_train, X_test, X_val, num_components=100)
    
    # Initialize and train the neural network with PCA-reduced data
    input_size_pca = X_train_pca.shape[1]
    model_pca = initialize_model(input_size=input_size_pca, hidden_size=20, output_size=10)
    train_model(model_pca, X_train_pca, y_train_one_hot, X_val_pca, y_val_one_hot, epochs=200, learning_rate=0.01)

    # Evaluation on Training Data (PCA)
    y_pred_train_pca = np.argmax(model_pca.forward(X_train_pca), axis=1)
    plot = confusion_matrix_plot(X_train_pca, y_true_train, y_pred_train_pca, split='Training (PCA)')
    plot.savefig('./plots/confusion_matrix_training_pca.png')

    # Evaluation on Validation Data (PCA)
    y_pred_val_pca = np.argmax(model_pca.forward(X_val_pca), axis=1)
    plot = confusion_matrix_plot(X_val_pca, y_true_val, y_pred_val_pca, split='Validation (PCA)')
    plot.savefig('./plots/confusion_matrix_validation_pca.png')

    # Evaluation on Testing Data (PCA)
    y_pred_test_pca = np.argmax(model_pca.forward(X_test_pca), axis=1)
    plot = confusion_matrix_plot(X_test_pca, y_true_test, y_pred_test_pca, split='Testing (PCA)')
    plot.savefig('./plots/confusion_matrix_testing_pca.png')

if __name__ == "__main__":
    main() 