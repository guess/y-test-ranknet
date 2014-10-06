import unittest, random
import ranknet, ffnet, activation

class Test(unittest.TestCase):

    seed = 1.0

    random.seed(seed)
    
    ninp = 10       # number of features
    nhid = 8        # number of hidden units
    nq = 5         # number of samples in a query
        
    # generated query data 
    query = [ [0, random.choice([0, 1])] + [random.random() for _ in xrange(ninp)] for _ in xrange(nq) ]
                 
    # neural network model
    model = ffnet.FFNet([ninp, nhid, 1], [activation.linear(), activation.tanh(), activation.sigmoid()])
        
    # random weights
    model.initw(1.0, seed)   
    w = model.getw()
    
    
    # ranknet sigma
    sigma = 1.0
            
    def test_s(self):
        self.assertEqual(1, ranknet.S(1, 0))
        self.assertEqual(0, ranknet.S(1, 1))
        self.assertEqual(0, ranknet.S(0, 0))
        self.assertEqual(-1, ranknet.S(0, 1))
       
    def test_cost(self):
               
        C = ranknet.cost(self.query, self.model, self.sigma)
        print C

    def test_gradient(self):
        
        # analytical gradient
        dbpr = ranknet.gradient(self.query, self.model, self.sigma)
        
        # numerical gradient
        dw = 1.e-5
        jw = 7
        
        w_u = self.w[:]
        w_u[jw] = w_u[jw] + dw
        self.model.setw(w_u)
        C_u = ranknet.cost(self.query, self.model, self.sigma)
        
        w_d = self.w[:]
        w_d[jw] = w_d[jw] - dw
        self.model.setw(w_d)
        C_d = ranknet.cost(self.query, self.model, self.sigma)
        
        dnum = (C_u[0] - C_d[0]) / dw / 2.0
        
        print dbpr[jw], dnum
    
    def test_gradient_2(self):
        
        # run tests
        ntest = 100
        ninp = 10
        nq = 20
        nhid = self.nhid
        nw = (ninp + 1) * nhid + (nhid + 1) * 1     # total number of weights
        dw = 1.e-6
               
        for j in xrange(ntest):
            
            # generate query data
            query = [ [0, random.choice([0, 1])] + [random.random() for _ in xrange(ninp)] for _ in xrange(nq) ]
            
            # get analytical gradient
            grad = ranknet.gradient(query, self.model, self.sigma)
                        
            # select weight at random
            jw = random.choice(xrange(nw))
                
            # numerical derivative                
            w_u = self.w[:]
            w_u[jw] = w_u[jw] + dw
            self.model.setw(w_u)
            C_u = ranknet.cost(query, self.model, self.sigma)
        
            w_d = self.w[:]
            w_d[jw] = w_d[jw] - dw
            self.model.setw(w_d)
            C_d = ranknet.cost(query, self.model, self.sigma)
        
            dnum = (C_u[0] - C_d[0]) / dw / 2.0
                
            # compare results
            self.assertAlmostEqual(grad[jw], dnum, 5, "Run %d: %e %e " % (j, grad[jw], dnum))

            print j, jw, grad[jw], dnum

if __name__ == "__main__":
    unittest.main()