from lin import zeros

class FFNet(object):

	"""
	Implementation of a feedforward neural network

	"""
	

	def __init__(self, layers, transf):
		"""
		Create a FFNet object with a given layer structure
		
		Arguments
		---------
		layers: list
			List of layer sizes [Ninput, Nhidden1, ..., NhiddenK, Noutput]
		transf: list
			List of layer transfer functions. {"lin", "sigm", or "tanh"}

		Examples
		--------
		>>> net = FFNet([8, 5, 3], ["lin", "tanh", "sigm"])
		"""
				
		self.W = []						# structured weights and biases
		self.layers = layers			# layer sizes
		self.nlayers = len(layers)      # number of layers  
		
		self.transf = transf			# transfer functions
		
		for j in xrange(0, self.nlayers-1):
			# matrix of weights between next and current layer
			self.W.append(zeros(layers[j]+1, layers[j+1])) # first row contains bias 
	
	def getw(self):
		"""
		Get current weights (and biases) as a flat list

		"""
		w = [];
		
		for matrix in self.W:
			for row in matrix:
				w = w + row
		
		return w
						

	def setw(self, w):
		"""
		Set weights (and biases) from a flat list
		
		"""
		count = 0;
		for j in xrange(0, self.nlayers-1):
			# matrix of weights between next and current layer
			for k in xrange(0, self.layers[j]+1):
				self.W[j][k] = w [ count : count + self.layers[j+1] ]
				count = count + self.layers[j+1]

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
		y: list
			List of neural network output values

		"""
		pass
	
	def jacobian(self, x):
		"""
		Compute jacobian of a network outputs with respect to weights at a given sample point 

		Arguments
		---------
		x: list
			List of input features of a single data point (single sample)	

		"""
		
		
		

	
		

	


	