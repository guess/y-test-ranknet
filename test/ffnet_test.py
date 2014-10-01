import ffnet, activation
import unittest, random

n = 3
k, q, m = 8, 5, 2

class FFNetTest(unittest.TestCase):
         
    def setUp(self):
        # super
        unittest.TestCase.setUp(self)
            
    def test_init(self):
   
        # create a simple network
        net = ffnet.FFNet([k, q, m], [])
                        
        # Test internal structured weight matrices
        self.assertEqual(n-1, len(net.W), "#1.1 Dimensions")
        self.assertEqual(q, len(net.W[0]), "#1.2 Dimensions")
        self.assertEqual(k+1, len(net.W[0][0]), "#1.3 Dimensions")
        self.assertEqual(m, len(net.W[1]), "#1.4 Dimensions")
        self.assertEqual(q+1, len(net.W[1][0]), "#1.5 Dimensions")
            
    def test_getw(self):

        # create a simple network
        net = ffnet.FFNet([k, q, m], [])

        # Test exported flat list
        self.assertEqual((k+1)*q + (q+1)*m, len(net.getw()), "#2 Wrong weight matrix dimensions")
        
    def test_setw(self):
        
        # create a simple network
        net = ffnet.FFNet([k, q, m], [])
        
        # set weights of the model: first layer with ones, second layer with twos
        net.setw([1] * (q*(k+1))  + [2] * m*(q+1))
        
        # Test updated weights
        self.assertEqual(1.0, net.W[0][0][0], "#3.1 Should be updated")
        self.assertEqual(2.0, net.W[1][5][1], "#3.2 Should be updated") 
        
    def test_apply(self):
            
        # create a simple network
        net = ffnet.FFNet([k, q, m], [activation.linear(), activation.sigmoid(), activation.sigmoid()])  

        # set weights
        w = net.getw()
        w = [-10] * len(w)
        net.setw(w)

        # some input
        x = [1] * 8
        z = net.apply(x)

        print z

    def test_backprop(self):
        
        # create a simple network
        net = ffnet.FFNet([k, q, m], [activation.linear(), activation.sigmoid(), activation.tanh()])  

        # set weights
        w = net.getw()
        w = [1] * len(w)
        net.setw(w)

        # some input
        x = [205.0] * 8
        z = net.apply(x)
        
        n = 45
        
        # backprop
        d = net.backprop([1.0/2, 0])
                       
        dw = 1.e-6
        w_u = w[:]
        w_u[n] = w_u[n] + dw
        net.setw(w_u)
        z1 = net.apply(x)
        
        w_d = w[:]
        w_d[n] = w_u[n] - dw
        net.setw(w_d)
        z2 = net.apply(x)
        
        print (z1[0] - z2[0])/dw/2
        print d[n]






if __name__ == '__main__':
    unittest.main()