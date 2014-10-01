import unittest, linfunc


class Test(unittest.TestCase):


    def test_gax(self):
        
        matrix = [[1, 2, 3], [4, 5, 6]]
        x = [1, 2]
        
        print linfunc.gax(matrix, x)
        
    def test_lgax(self):
        
        matrix = [[1, 2, 3], [4, 5, 6]]
        x = [2, 2, 2]
        
        print linfunc.lgax(x, matrix)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()