import ffnet, activation
import la
import unittest, random

n = 3
k, q, m = 10, 5, 1

class Test(unittest.TestCase):
         
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
        net = ffnet.FFNet([k, q, m], [activation.linear(), activation.tanh(), activation.sigmoid()])  

        # set weights
        w = net.getw()
        w = [-10] * len(w)
        net.setw(w)

        # some input
        x = [1] * 8
        net.apply(x)

    def test_backprop(self):
        
        #===================================
        # Gradient check
        #===================================
        
        # create a simple network
        net = ffnet.FFNet([k, q, m], [activation.linear(), activation.sigmoid(1.2), activation.tanh(2.0, 3.0/2.0)])  

        # set random weights
        wscale = 3.0
        w = net.getw()
        w = [random.uniform(-wscale, wscale) for _ in w]
        net.setw(w)

        # run tests
        ntest = 1000
               
        for j in xrange(ntest):
            
            # generate random input vector
            xscale = 5.0
            x = [random.uniform(-xscale, xscale) for _ in xrange(k)]
            
            # propagate forward
            net.apply(x)

            # select a weight at random
            N = (k+1)*q + (q+1)*m           # total number of weights
            nw = random.randint(0, N-1)       # select one weight at random
                        
            # backprop
            for jm in xrange(m):
                
                # compute derivative of output with 
                # respect to input using back propagation
                dbpr = net.backprop(la.unit(m, jm))    # initial delta is just a unit vector
                
                # numerical derivative           
                dw = 1.e-6
                
                w_u = w[:]
                w_u[nw] = w_u[nw] + dw
                net.setw(w_u)
                z1 = net.apply(x)
                                
                w_d = w[:]
                w_d[nw] = w_d[nw] - dw
                net.setw(w_d)
                z2 = net.apply(x)
                
                dnum = (z1[jm] - z2[jm]) / dw / 2
                
                # compare results
                self.assertAlmostEqual(dbpr[nw], dnum, 5, "Run %d, output %d: %e %e " % (j, jm, dbpr[nw], dnum))

                print j, nw, jm, dbpr[nw], dnum


if __name__ == '__main__':
    unittest.main()