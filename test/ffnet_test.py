import ffnet
import unittest

n = 3
k, l, m = 8, 5, 2

class FFNetTest(unittest.TestCase):
         
    def setUp(self):
        # super
        unittest.TestCase.setUp(self)
            
    def test_init(self):
   
        # create a simple network
        net = ffnet.FFNet([k, l, m], [])
                        
        # Test internal structured weight matrices
        self.assertEqual(n-1, len(net.W), "#1.1 Dimensions")
        self.assertEqual(k+1, len(net.W[0]), "#1.2 Dimensions")
        self.assertEqual(l, len(net.W[0][0]), "#1.3 Dimensions")
        self.assertEqual(l+1, len(net.W[1]), "#1.4 Dimensions")
        self.assertEqual(m, len(net.W[1][0]), "#1.5 Dimensions")
            
    def test_getw(self):

        # create a simple network
        net = ffnet.FFNet([k, l, m], [])

        # Test exported flat list
        self.assertEqual(l*(k+1) + m*(l+1), len(net.getw()), "#2 Wrong weight matrix dimensions")
        
    def test_setw(self):
        
        # create a simple network
        net = ffnet.FFNet([k, l, m], [])
        
        # set weights of the model: first layer with ones, second layer with twos
        net.setw([1] * (l*(k+1))  + [2] * m*(l+1))
        
        # Test updated weights
        self.assertEqual(1, net.W[0][0][0], "#3.1 Should be updated")
        self.assertEqual(2, net.W[1][5][1], "#3.2 Should be updated")
        print net.W        

if __name__ == '__main__':
    unittest.main()