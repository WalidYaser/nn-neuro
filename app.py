import numpy as np

def tanh(x):
    return np.tanh(x)

input_neurons = 2  
hidden_neurons = 2  
output_neurons = 1  

np.random.seed(42)  
W1 = np.random.uniform(-0.5, 0.5, (hidden_neurons, input_neurons))  
W2 = np.random.uniform(-0.5, 0.5, (output_neurons, hidden_neurons))  

b1 = np.array([0.5, 0.7])  
b2 = 0.7  

X = np.random.uniform(-1, 1, (input_neurons,))  

hidden_input = np.dot(W1, X) + b1
hidden_output = tanh(hidden_input)

final_input = np.dot(W2, hidden_output) + b2
final_output = tanh(final_input)

print("Input:", X)
print("Hidden Layer Output:", hidden_output)
print("Final Output:", final_output)