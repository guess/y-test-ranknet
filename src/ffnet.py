"""
Feed-forward neural network
===========================

"""

import la
from random import Random

class FFNet(object):

	"""
	Implementation of a feedforward neural network

	"""
	

	def __init__(self, layers, activf):
		"""
		Create a FFNet object with a given layer structure
		
		Arguments
		---------
		layers: list
			List of layer sizes [Ninput, Nhidden1, ..., NhiddenK, Noutput]
		activf: list
			List of layer transfer functions objects

		Examples
		--------
		>>> net = FFNet([8, 5, 3], [activation.linear(), activation.sigmoid(), activation.tanh(1.8, 3./2)])
		"""
				
		self.W = []						# structured weights and biases
		self.layers = layers			# layer sizes
		self.nlayers = len(layers)      # number of layers  
		
		self.activf = activf			# transfer functions
		
		self.y = []						# layer inputs
		self.z = []						# layer activations
		
		for j in xrange(0, self.nlayers-1):
			
			# a column of weight matrix including bias
			column = [0] * (layers[j+1])	
	
			# repeat for each node of this layer + bias
			m = [column] * (layers[j] + 1)
				
			# append to list of weights
			self.W.append(m)
	
	def initw(self, scale = 1.0, seed = None):
		"""
		Random initialization of network weights. Biases are initialized to 0. 
		"""
		if seed is not None:
			rand = Random(seed)
		else:
			rand = Random()
		
		w = [];
			
		for matrix in self.W:
			# initialize bias			
			w = w + [0.0] * len(matrix[0])
			# initialize weights
			for jc in xrange(1, len(matrix)):
				w = w + [rand.gauss(0.0, scale) for _ in xrange(len(matrix[jc]))]
		
		# update weights of the network
		self.setw(w)
		
	def getw(self):
		"""
		Get current weights (and biases) as a flat list
		"""
		w = [];
		
		for matrix in self.W:
			for column in matrix:
				w = w + column
		
		return w			

	def setw(self, w):
		"""
		Set weights (and biases) from a flat list
		"""
		count = 0;
		for j in xrange(0, self.nlayers-1):
			m = []
			# matrix of weights between next and current layer
			for _ in xrange(0, self.layers[j]+1):
				column = w[count : count + self.layers[j+1]] 
				m.append(column)
				count = count + self.layers[j+1]
			# put into list of matrices
			self.W[j] = m
			
	def apply(self, x):
		"""
		Compute output of neural network for a single data sample 
		with current weights.

		Arguments
		---------
		x: list
			List of input features of a single data point (single sample)

		Returns
		-------
		z: list
			Neural network output values

		"""
		
		# clear working memory
		self.y = []
		self.z = []
		
		# input to the first layer 
		self.y += [x]
			
		# propagate forward			
		for j in xrange(0, self.nlayers-1):
			# current layer activation 
			self.z += [map( self.activf[j].f, self.y[j] )]
			# add bias 
			z_ = [1] + self.z[j]
			# next layer input
			self.y += [la.gax(self.W[j], z_)]
			
		# output layer activation
		j = self.nlayers-1
		self.z += [map( self.activf[j].f, self.y[j] )]
		
		# return the output activation
		return self.z[self.nlayers - 1]
	
	def backprop(self, d):
		"""
		Compute gradients  

		Arguments
		---------
		x: list
			List of input features of a single data point (single sample)
		
		d: list of backpropagating errors	

		"""
		
		dy = [[]] * self.nlayers
		dz = [[]] * self.nlayers
		dW = [[]] * (self.nlayers - 1)
		
		k = self.nlayers - 1
		dz[k] = d
		dy[k] = la.vmul(d, map( self.activf[k].df, self.z[k] ))
		
		for j in xrange(k, 0, -1):
			# compute derivatives with respect to biases and weights
			dW[j-1] = [dy[j]] + la.outer(dy[j], self.z[j-1])
			# backpropagate connections
			dz[j-1] = la.lgax(dy[j], self.W[j-1])
			# remove bias
			dz_ = dz[j-1][1:]	
			# backpropagate activation
			dy[j-1] = la.vmul(dz_, map( self.activf[j-1].df, self.z[j-1] ))
			
		
		# expand derivatives		
		dw = [];
		for matrix in dW:
			for column in matrix:
				dw = dw + column
		# done
		return dw
	


	