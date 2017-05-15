import math, random

def sigmoid(vec):
	return [1.0 / (1.0 + math.e ** -elem) for elem in vec]

def softmax(vec):
	exp = [math.e ** elem for elem in vec]
	total = sum(exp)
	return [elem / total for elem in exp]

class NeuralNetworkLayer:
	def __init__(self, input_size, output_size, activation):
		self.input_size = input_size
		self.output_size = output_size
		self.activation = activation
		self.weights = [[random.uniform(0, 1) for x in range(0, input_size)] for y in range(0, output_size)]
		self.bias = [random.uniform(0, 1) for y in range(0, output_size)]

	def apply(self, inputs):
		outputs = []
		for i in range(self.output_size):
			sum = 0
			for j in range(self.input_size):
				sum += self.weights[i][j] * inputs[j]
			sum += self.bias[i]
			outputs.append(sum)
		return self.activation(outputs)


inputs = [1, 2, 3]
hidden_layer = NeuralNetworkLayer(3, 4, sigmoid)
hidden_layer2 = NeuralNetworkLayer(4, 4, sigmoid)
output_layer = NeuralNetworkLayer(4, 2, softmax)
outputs = output_layer.apply(hidden_layer2.apply(hidden_layer.apply(inputs)))
print("Input: {}".format(inputs))
print("Hidden layer:\n  Weights: {}\n  Bias: {}".format(hidden_layer.weights, hidden_layer.bias))
print("Output: {}".format(outputs))
