import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of sigmoid activation function
        return x * (1 - x)

    def forward(self, X):
        # Forward pass
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predictions = self.sigmoid(self.output_layer_input)
        return self.predictions

    def backward(self, X, y, learning_rate=0.01):
        # Backward pass
        error = y - self.predictions
        output_delta = error * self.sigmoid_derivative(self.predictions)
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, X_val=None, y_val=None, epochs=100, learning_rate=0.001):
        """Trains the neural network using the specified data and hyperparameters.

        Args:
            X (np.array): Training data.
            y (np.array): Training labels.
            X_val (np.array, optional): Validation data. Defaults to None.
            y_val (np.array, optional): Validation labels. Defaults to None.
            epochs (int, optional): Number of epochs to train for. Defaults to 100.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
        """
        print("Epoch\tTraining Loss\tEvaluation Loss")
        for epoch in range(epochs):
            # Forward pass
            self.predictions = self.forward(X)

            # Backward pass and parameter updates
            self.backward(X, y, learning_rate)

            # Print loss for training and validation data every 5 epochs
            if epoch % 5 == 0:
                train_loss = np.mean(np.square(y - self.predictions) / 2)
                if X_val is not None and y_val is not None:
                    eval_predictions = self.forward(X_val)
                    eval_loss = np.mean(np.square(y_val - eval_predictions) / 2)
                    print(f"{epoch}\t{train_loss:.3f}\t{eval_loss:.3f}")
                else:
                    print(f"{epoch}\t{train_loss:.3f}") 