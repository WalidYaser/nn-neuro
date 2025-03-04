import numpy as np


inputs = np.array([0.05, 0.10])
targets = np.array([0.01, 0.99])
learning_rate = 0.5


weights = {
    'hidden': np.array([[0.15, 0.20], [0.25, 0.30]]),
    'output': np.array([[0.40, 0.45], [0.50, 0.55]])
}

biases = {
    'hidden': 0.35,
    'output': 0.60
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def forward_pass(x):

    net_h = np.dot(x, weights['hidden'].T) + biases['hidden']
    h = sigmoid(net_h)
    
    
    net_o = np.dot(h, weights['output'].T) + biases['output']
    o = sigmoid(net_o)
    
    return h, o


def backward_pass(x, h, o, y):

    error_grad_o = -(y - o) * sigmoid_derivative(o)
    delta_w_output = np.outer(error_grad_o, h)
    

    error_grad_h = np.dot(error_grad_o, weights['output']) * sigmoid_derivative(h)
    delta_w_hidden = np.outer(error_grad_h, x)
    
    return delta_w_hidden, delta_w_output


h, o = forward_pass(inputs)
error = 0.5 * np.sum((targets - o) ** 2)
print(f"New error before update: {error:.6f}")

delta_hidden, delta_output = backward_pass(inputs, h, o, targets)

weights['hidden'] -= learning_rate * delta_hidden
weights['output'] -= learning_rate * delta_output

h_new, o_new = forward_pass(inputs)
error_new = 0.5 * np.sum((targets - o_new) ** 2)
print(f"New error after update: {error_new:.6f}")