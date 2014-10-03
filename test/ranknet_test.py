import unittest, random
import ranknet, ffnet, activation

class Test(unittest.TestCase):

    random.seed(102.0)
    
    ninp = 10       # number of features
    nhid = 8        # number of hidden units
    nq = 5         # number of samples in a query
        
    # generated query data 
    query = [ [0, random.choice([0, 1])] + [random.random() for _ in xrange(ninp)] for _ in xrange(nq) ]
    
    print query[0][1], query[1][1], query[2][1]
              
    # neural network model
    model = ffnet.FFNet([ninp, nhid, 1], [activation.linear(), activation.tanh(), activation.sigmoid()])
        
    # random weights
    w = model.getw();
    w = [random.uniform(-1.6, 1.6) for _ in w]
    model.setw(w)    
            
    def test_s(self):
        self.assertEqual(1, ranknet.s(1, 0))
        self.assertEqual(0, ranknet.s(1, 1))
        self.assertEqual(0, ranknet.s(0, 0))
        self.assertEqual(-1, ranknet.s(0, 1))
       
    def test_cost(self):
               
        C = ranknet.cost(self.query, self.model)
        print C

    def test_gradient(self):
        
        # analytical gradient
        dbpr = ranknet.gradient(self.query, self.model)
        
        # numerical gradient
        dw = 1.e-5
        jw = 7
        
        w_u = self.w[:]
        w_u[jw] = w_u[jw] + dw
        self.model.setw(w_u)
        C_u = ranknet.cost(self.query, self.model)
        
        w_d = self.w[:]
        w_d[jw] = w_d[jw] - dw
        self.model.setw(w_d)
        C_d = ranknet.cost(self.query, self.model)
        
        dnum = (C_u - C_d) / dw / 2.0
        
        print dbpr[jw], dnum
        

if __name__ == "__main__":
    unittest.main()