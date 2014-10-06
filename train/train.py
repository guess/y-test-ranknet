import ffnet, activation, ranknet, la
import random

def train(data):
    
    print "Total number of queries: ", len(data)
  
    # ranknet sigma
    sigma = 1.0
  
    # network structure
    layers = [13, 1]
    activf = [activation.linear(),  activation.sigmoid()] # activation.tanh(1.75, 3./2.),
    net = ffnet.FFNet(layers, activf)
    net.initw(0.01)               
    
    # random permutation of data
    perm = range(len(data))
    random.shuffle(perm)
    n = len(perm)
    
    # learning rate
    rate = 1.e-3
    
    # number of epochs
    maxepoch = 500000
            
    # training
    for je in xrange(maxepoch):
        
        # take next query
        jq = je % n
        query = data[jq]
        
        # compute cost and estimates
        C = ranknet.cost(query, net, sigma)
        
        # print
        if jq == 95:
            print je, jq, [r[1] for r in query],  C[0], C[1], C[2]
            
        
        # compute gradients
        g = ranknet.gradient(query, net, sigma)
        
        # update weights
        w = la.vsum(net.getw(), la.sax(-rate, g))
        net.setw(w)
                     
        
        
        
        
        
        
        
        
        
        
        
        