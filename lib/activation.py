from abc import abstractmethod
import math

"""
Activation functions
"""

class ActivationFunction(object):
    
    @abstractmethod
    def f(self, x): 
        """
        Activation function value at x
        """
        pass
    
    @abstractmethod
    def df(self, f): 
        """
        Activation function derivative at x, computed from function value 
        """
        pass
    

class sigmoid(ActivationFunction):
    """
    Compute sigmoid f(x) = a / ( 1 + exp(-x/s) ) and its derivative
    """
    def __init__(self, a = 1.0, s = 1.0):
        self.a = a
        self.s = s
    
    def f(self, x):
        # non-overflowing sigmoid 
        return self.a * ( 1.0 + math.tanh(x/self.s/2.0) ) / 2.0
    
    def df(self, f):
        return  f * (self.a - f) / self.a / self.s
    
class linear(ActivationFunction):
           
    def f(self, x): 
        return x 
    
    def df(self, f):
        return 1.0
    
class tanh(ActivationFunction):
    
    def __init__(self, a = 1.0, s = 1.0):
        self.a = a
        self.s = s
    
    def f(self, x):
        return self.a * math.tanh( x / self.s )
    
    def df(self, f):
        return ( self.a*self.a - f*f  ) / self.a / self.s
    
        