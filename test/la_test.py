import unittest
import la


class Test(unittest.TestCase):


    def test_gax(self):
        
        matrix = [[1, 2, 3], [4, 5, 6]]
        x = [1, 2]
        
        print la.gax(matrix, x)
        
    def test_lgax(self):
        
        matrix = [[1, 2, 3], [4, 5, 6]]
        x = [2, 2, 2]
        
        print la.lgax(x, matrix)

    def test_unit(self):
        
        u = la.unit(6, 2)
        
        print u

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()