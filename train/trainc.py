import ffnet, activation, ranknet, la
import random

def train(data, options):
    
    """
    Train a committee of ranknet models 
    """
   
    print "Total number of queries: ", len(data)
  
    # ranknet sigma
    sigma = 2.0
  
    # network structure
    layers = [13, 8, 1]
    activf = [activation.linear(), activation.tanh(), activation.sigmoid()] #  ,  activation.tanh(1.75, 3./2.),
    net = ffnet.FFNet(layers, activf)
    net.initw(0.1)               
    
    # random permutation of data
    perm = range(len(data))
    random.shuffle(perm)
    n = len(perm)
    
    # validation 
    nv = 100                # number of validation samples
    jvalid = perm[0:nv]     # validation data
    epvalid = 500           # epochs between validation 
    
    # training
    ntr = n - nv
    jtrain = perm[nv:]
    
    # learning rate
    rate = 1.e-4
    
    # number of epochs
    maxepoch = 10000000
    
    # report file
    fname = "report" + "1" + ".txt"
    f = open(fname, "w+")
            
    # training
    for je in xrange(maxepoch):
        
        # validation
        if je % epvalid == 0:
            # compute validation error
            C = 0.0
            for j in jvalid:
                c = ranknet.cost(data[j], net, sigma)
                C += c[0]
           
            # print
            print je, C, net.getw()
            
            # write to report file
            sw = str(net.getw()).replace("[", "").replace("]", "").replace(",", "")
            f.write("%d %e %e %s\n" % (je, rate, C, sw))
            
        # take next query
        jq = je % ntr
        query = data[jtrain[jq]]
                    
        # compute gradients
        g = ranknet.gradient(query, net, sigma)
        
        # update weights
        w = la.vsum(net.getw(), la.sax(-rate, g))
        net.setw(w)
     
    print "Training stopped"
    print "Final weights: ", net.getw()
    
    # close file 
    f.close() 
        
def test(data, net, sigma):     
    
    Q = [[0, 0], [0, 0]]            # confusion matrix
    
    for query in data:
        # compute predictions
        C = ranknet.cost(query, net, sigma)
        for j in xrange(len(query)):
            k = query[j][1]
            q = int(C[1][j] > 0.5)
            Q[q][k] += 1
                
    return Q
        
        
        
        
        
        
        
        