import unittest, random
import ranknet, ffnet, activation, la

class Test(unittest.TestCase):

    seed = 1.0

    random.seed(seed)
    
    ninp = 10       # number of features
    nhid = 8        # number of hidden units
    nq = 5         # number of samples in a query
        
    # generated query data 
    query = [ [0, random.choice([0, 1])] + [random.random() for _ in xrange(ninp)] for _ in xrange(nq) ]
                 
    # neural network model
    model = ffnet.FFNet([ninp, nhid, 1], [activation.linear(), activation.tanh(1.75, 3./2.), activation.sigmoid()]) 
        
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
            #self.assertAlmostEqual(grad[jw], dnum, 5, "Run %d: %e %e " % (j, grad[jw], dnum))

            print j, jw, grad[jw], dnum
            
    def xtest_training_1(self):
        
        # train on a single query
        
        nepoch = 10000    # number of training epochs
        rate = 0.1        # learning rate
        nprint = 1000     # print frequency
                
        for je in xrange(nepoch):
            
            # compute current cost and estimations
            C = ranknet.cost(self.query, self.model, self.sigma)
            if je % nprint == 0:
                print je, C[0], C[1], C[2]
                print "w:", self.model.getw() 
            # compute gradients
            g = ranknet.gradient(self.query, self.model, self.sigma)
        
            # update weights
            w = la.vsum( self.model.getw(), la.sax(-rate, g) )
            self.model.setw(w)
        
    def xtest_training_2(self): 
        
        # train on several queries
        data = []
        d = range(10)
        for j in d:
            data.append( [ [j, random.choice([0, 1])] + [random.random() for _ in xrange(self.ninp)] for _ in xrange(self.nq) ] )
        
        print data
                
        nepoch = 10000    # number of training epochs
        rate = 0.1        # learning rate
        nprint = 1000     # print frequency
        
        # compute current cost and estimations
        for je in xrange(nepoch):
            
            # select training sample at random
            jq = random.choice(d)   
            
            if je % nprint == 0:
                
                # compute cost of a first sample
                C = ranknet.cost(data[0], self.model, self.sigma)
                
                print je, C[0], C[1], C[2]
                print "w:", self.model.getw() 
            
            # compute gradients
            g = ranknet.gradient(data[jq], self.model, self.sigma)
        
            # update weights
            w = la.vsum( self.model.getw(), la.sax(-rate, g) )
            self.model.setw(w)
    
        # final report
        for query in data:
            print "Query: ", query[0][0]
            C = ranknet.cost(query, self.model, self.sigma)
            for j in xrange(len(query)):
                print query[j][1], C[1][j]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    

if __name__ == "__main__":
    unittest.main()