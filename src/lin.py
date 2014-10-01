"""
Basic linear algebra data structures and utilities.

Note: The implementation is inefficient and intended for educational purposes!
In a production implementation this should be replaced by data structures and 
algorithms from Numpy or similar package.  

"""

import collections

class vector(collections.MutableSequence):

    """
    Fixed size vector backed by a list. All values are stored as floats.
    
    """
    
    def __init__(self, elements):
        # wrap a list
        # copy from argument and convert to float
        self.list = map(float, elements)
        self.size = len(self.list)   
    
    def __getitem__(self, index):
        # delegate to list
        return self.list[index]
      
    def __setitem__(self, index, value):
        # convert to float
        self.list[index] = float(value)
        
    def __delitem__(self, index):
        # not supported
        pass
    
    def __len__(self):
        # delegate to list
        return len(self.list)
    
    def insert(self):
        # not supported
        pass
    
    def __repr__(self):
        # customized string representation
        return "v" + str(self.list)
    
    def __add__(self, that):
        
        """
        Compute element-wise sum of two vectors. Operands should 
        have the same length. 
        """
                
        # vectors should be of the same length
        assert self.size == len(that), "Vectors must be of the same length"
        
        # create new vector
        result = vector(self.list);
        
        # sum components
        for j in xrange(0, len(self)):
            result[j] = self[j] + that[j]
                
        return result
    
    def __mul__(self, x):
        
        """ 
        Multiply a vector by a scalar.
        """
                                
        # create new vector
        result = vector(self.list);
        
        # sum components
        for j in xrange(0, len(self)):
            result[j] = self[j] * x
                
        return result
    
    def __rmul__(self, x):
        
        """
        Multiply a scalar by a vector.
        """
        
        # swap operands
        return self.__mul__(x)
    
    def __div__(self, x):
        
        """
        Divide a vector by a scalar
        """        
        
        # delegate to multiplication
        return self.__mul__(1.0/x)
    
    
    def inner(self, that):
        
        """
        Dot product of two vectors
        """
        
        assert self.size == len(that), "Vectors must be of the same size"
        
        result = 0
        
        for j in xrange(0, self.size):
            result += self.list[j]*that[j]
        
        return result
    
    def outer(self, that):
        
        """
        Outer product of two vectors
        
        Returns
        -------
        A matrix of size len(self) by len(that)
        
        """
        # construct a matrix by sequentially multiplying
        # `this` vector by each element of `that` vector
        return matrix( map(lambda x: self*x, that) )
     
    def to_list(self):
        # return vector as list
        return self.list[:]   


class matrix(collections.MutableSequence):

    """
    Fixed size matrix as collection of columns. Each column is a vector.
    
    """
    
    def __init__(self, columns):
        self.list = []
        self.m = len(columns)               # number of columns
        self.n = len(columns[0])            # number of rows
        for c in columns:
            assert len(c) == self.n, "Columns must be of the same length"
            self.list.append(vector(c))     # convert to vector
        
    def __getitem__(self, index):
        """
        Get element of a matrix, or a whole column
        
        Examples
        --------
        >>> m = matrix([[1, 2, 3],[0, 0, 0]])
        >>> m[0, 1] 
        2
        >>> m[1]
        [0, 0, 0]       
        """
        if isinstance(index, tuple):
            return self.list[index[1]][index[0]]
        else:
            return self.list[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            # set one value
            self.list[index[1]][index[0]] = value
        else:
            # set a whole column
            assert(len(value) == self.n)
            self.list[index] = vector(value)

    def __delitem__(self, index):
        # not supported
        pass
    
    def __len__(self):
        # return number of columns
        return len(self.list)
    
    def insert(self):
        # not supported
        pass

    def __repr__(self):
        return "m" + str(self.list)

    def __mul__(self, that):
        """
        Multiply matrix by a vector or by a matrix.
        Using column-wise version of matrix multiplication.
        See: G. Golub, C. Van Loan, 1.1.11 and 1.1.15 
        """
        
        if isinstance(that, vector):
            assert self.m == that.size, "Matrix dimensions must agree"             
            z = vector([0] * self.n)    # initial result is zero vector
            for j in xrange(0, self.m):
                z = z + self[j] * that[j]
            return z
        if isinstance(that, matrix):
            # multiply `this` matrix by each column of `that` matrix
            return matrix(map(lambda c: self * c, that))
