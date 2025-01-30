from neural_network import NeuralNetwork

def initialize_model(input_size, hidden_size=20, output_size=10):
    """Initializes the Neural Network model.

    Args:
        input_size (int): Number of input features.
        hidden_size (int, optional): Number of hidden neurons. Defaults to 20.
        output_size (int, optional): Number of output classes. Defaults to 10.

    Returns:
        NeuralNetwork: Initialized neural network model.
    """
    model = NeuralNetwork(input_size, hidden_size, output_size)
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, learning_rate=0.01):
    """Trains the neural network model.

    Args:
        model (NeuralNetwork): The neural network model to train.
        X_train (np.array): Training data.
        y_train (np.array): One-hot encoded training labels.
        X_val (np.array): Validation data.
        y_val (np.array): One-hot encoded validation labels.
        epochs (int, optional): Number of training epochs. Defaults to 200.
        learning_rate (float, optional): Learning rate. Defaults to 0.01.
    """
    model.train(X_train, y_train, X_val, y_val, epochs=epochs, learning_rate=learning_rate) 