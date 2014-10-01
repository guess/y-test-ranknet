from lin import vector, matrix
import unittest

class UtilsTest(unittest.TestCase):
    
    def test_vector_add(self):
        v = vector([1, 2, 3])
        print v
        
        v = v + vector([3, 2, 1])
        print v
        
        v = v*5
        print v
        
        v = 5*v
        print v
        
        v = v/5
        print v
        
        print map(float, [1, 2])

        m = matrix([[1, 2], [3, 4]])
        
        print m
        
        print m[1, 1]
        
        v1 = vector([1, 2])
        v2 = vector([0, 1])
        
        m = v1.outer(v2)
        
        print m
        
        m = v2.outer(v1)
        
        print m
        
        m = matrix([[1, 2, 3], [1, 2, 3]])
        
        print m
        
        v3 = m * vector([1, 2])
        print v3
        
        m2 = matrix([[1, 3], [1, 3]])

        m3 = m*m2
        
        print m3
        
        
if __name__ == '__main__':
    pass