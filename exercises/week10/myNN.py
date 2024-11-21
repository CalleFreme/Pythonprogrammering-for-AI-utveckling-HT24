import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (X: input, y: output)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
y = np.array([[0], [1], [1], [0]])              # XOR outputs

# Initialize weights and biases
np.random.seed(0)
weights_input_hidden = np.random.rand(2, 2)  # 2 input nodes, 2 hidden nodes
weights_hidden_output = np.random.rand(2, 1) # 2 hidden nodes, 1 output node
bias_hidden = np.random.rand(1, 2)
bias_output = np.random.rand(1, 1)
learning_rate = 0.1

# Training loop
for epoch in range(10000):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute error
    error = y - predicted_output
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    d_hidden_layer = np.dot(d_predicted_output, weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += np.dot(hidden_layer_output.T, d_predicted_output) * learning_rate
    weights_input_hidden += np.dot(X.T, d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("Final predicted output:")
print(predicted_output)
