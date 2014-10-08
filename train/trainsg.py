import ranknet, la
import random

class options(object):
    
    def __init__(self):
        self.sigma = 1.0                    # ranknet sigma
        self.rate = 1.e-3                   # learning rate
        self.nvalid = 100                   # number of samples in validation set
        self.nepval = 500                   # number of epochs between validation and printing
        self.maxepoch = int(1.e+8)          # maximum number of iterations
        self.maxfail = 100                  # maximum number of validation fails
    
    def __repr__(self):
        return "options sigma: %e rate: %e nvalid: %d nepval: %d maxepoch: %d maxfail: %d" % (self.sigma, self.rate, self.nvalid, self.nepval, self.maxepoch, self.maxfail)

def train(data, opts, net, writefcn):
    """
    Stochastic gradient training of ranknet model 
    """ 
    
    # random permutation of data
    perm = range(len(data))
    random.shuffle(perm)
        
    jvalid = perm[0:opts.nvalid]    # validation data index                         
    jtrain = perm[opts.nvalid:]     # training data index
    
    nfail = 0               # current number of validation fails
    mincost = 1.e+100       # current known minimal validation cost
    wbest = net.getw()      # weights for minimal validation error
    
    print "Start stochastic gradient descent training, number of queries: ", len(data)
    print str(opts)
    print str(net)
            
    # stochastic gradient training
    for je in xrange(opts.maxepoch):
        
        # validation
        if je % opts.nepval == 0:
            
            # compute validation cost
            C = 0.0
            for j in jvalid:
                c = ranknet.cost(data[j], net, opts.sigma)
                C += c[0]
                
            # update best estimates
            if C < mincost: 
                mincost = C
                wbest = net.getw()
            else: 
                nfail += 1
            
            # check stopping criteria             
            if opts.maxfail > 0 and nfail >= opts.maxfail:
                break    
            
            # print
            print je, nfail, C, net.getw()
            
            # write to report file
            sw = str(net.getw()).replace("[", "").replace("]", "").replace(",", "")
            writefcn("%d %d %e %s\n" % (je, nfail, C, sw))
            
        # next training query
        jq = je % len(jtrain)
        query = data[jtrain[jq]]
                    
        # compute gradients
        g = ranknet.gradient(query, net, opts.sigma)
        
        # update weights
        w = la.vsum(net.getw(), la.sax(-opts.rate, g))
        net.setw(w)
     
    print "Training stopped"
    print "Final cost: ", mincost
    print "Final weights: ", wbest
    
    # return updated model
    net.setw(wbest)
    return net    
        