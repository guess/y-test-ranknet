import ffnet, activation, ranknet, la
import math, random

def train(data):
    
    random.seed(100.0)
    
    # network structure
    layers = [13, 8, 1]
    activf = [activation.linear(), activation.tanh(1.75, 3./2.), activation.linear()] 
    net = ffnet.FFNet(layers, activf)
    net.initw(0.1, 100.0)               # seed

    # total number of queries in input data 
    n = len(data)
    
    # generate random permutation of training data 
    perm = range(n)
    random.shuffle(perm)
    
    # use fixed number of queries for validation
    nvalid = 100
    
    # number of training epochs
    nepoch = 500000
    
    # learning rate
    rate = 1.e-5
    
    # print frequency
    nprint = 100
        
    # stochastic gradient descent
    for je in xrange(nepoch):
    
        # reporting
        if je % nprint == 0:
            
            # compute validation cost
            C = 0.0
            for jv in xrange(nvalid):
                C = C + ranknet.cost(data[perm[jv]], net)
            
            # explicit review of one element
            for r in test(data[perm[0]], net):
                print r
            
            # print report string
            print "epoch: %d validation cost: %e" % (je, C)
            
        # take next training query
        jq = random.choice(perm[nvalid:])
        
        # compute gradients
        g = ranknet.gradient(data[jq], net)
    
        # update weights
        w = la.vsum(net.getw(), la.sax(-rate, g))
        net.setw(w)
    
        
def test(query, model):
    
    # compute scores
    scores = [];
    
    for u in query:
        scores += model.apply(u[2:])
    
    # probabilities
    res = []
    for j in xrange(0, len(query)-1):
        for k in xrange(j+1, len(query)):
            res.append([query[j][0], query[j][1], query[k][1], scores[j], scores[k], 1.0 / ( 1.0 + math.exp( - 1.0 * (scores[j] - scores[k])) )])
          
    return res    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        