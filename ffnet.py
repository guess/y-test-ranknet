import math

class FFNet

	def __init__(self, layers, transf, errf):
		"""
		Constructor: creates FFNet object with a specified layer structure
		Arguments:
			layers - list of layers sizes, i.e. [ninput, nhidden1, ..., nhiddenK, noutput]
			transf - list of layer transfer functions, i.e. ["lin", "sigm", "tanh", "sigm"] 
			errf - error function, i.e. ["sse", "entropy"]
		"""
		return "TODO"

	def apply(self, x):
		return "TODO"
	
	def gradient(self, x, y):
		return "TODO"

	def setmodel(self, W)
		
# ------ Transfer functions ------

def sigm(x):
# ------------------------------------------------
	'''Standard sigmoid function'''
	return 1/(1 + math.exp(-x))
#-------------------------------------------------
# 

def dsigm(x):
#-------------------------------------------------
	return "TODO"
#-------------------------------------------------
		
	


	