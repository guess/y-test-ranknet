import ranknet, la, opt
import random

class options(object):
    
    def __init__(self):
        self.sigma = 1.0                    # ranknet sigma
        self.nvalid = 100                   # number of samples in validation set
        self.maxepoch = int(1.e+8)          # maximum number of iterations
        self.maxfail = 100                  # maximum number of validation fails
    
    def __repr__(self):
        return "options sigma: %e nvalid: %d maxepoch: %d maxfail: %d" % (self.sigma, self.nvalid, self.maxepoch, self.maxfail)
        
def train(data, opts, net, writefcn):
    """
    Batch training of ranknet model using RProp
    """
         
    # random permutation of data
    perm = range(len(data))
    random.shuffle(perm)
        
    jvalid = perm[0:opts.nvalid]    # validation data index                         
    jtrain = perm[opts.nvalid:]     # training data index
       
    nfail = 0               # current number of validation fails
    mincost = 1.e+100       # current known minimal validation cost
    wbest = net.getw()      # weights for minimal validation error
    
    # write out options and initial network
    writefcn(str(opts) + "\n")
    writefcn(str(net) + "\n")    
    
    # initialize RProp working memory
    rpropmem = ( [1.e-5] * len(net.getw()), [ 1 ] * len(net.getw()) )
    
    print "Start batch training, number of queries: ", len(data)
    print str(opts)    
            
    # training iterations
    for je in xrange(opts.maxepoch):
                
        # validation cost
        vcost = 0.0                     
        for j in jvalid:
            c = ranknet.cost(data[j], net, opts.sigma)
            vcost += c[0]
        
        # update best estimates
        if vcost < mincost: 
            mincost = vcost
            wbest = net.getw()
        else: 
            nfail += 1
        
        # check stopping criteria                
        if opts.maxfail > 0 and nfail >= opts.maxfail:
            break
        
        # reset accumulators
        tcost = 0.0                     # training cost
        G = [0] * len(net.getw())       # accumulated gradient
        
        # batch training 
        for jt in jtrain:
            # take next training query
            query = data[jt]
            
            # compute cost
            c = ranknet.cost(query, net, opts.sigma)
            tcost += c[0]
                        
            # compute gradient 
            g = ranknet.gradient(query, net, opts.sigma)
        
            # update batch gradient 
            G = la.vsum(G, g)
        
        # print to screen
        print je, nfail, tcost, vcost, net.getw()
        
        # write out
        sw = str(net.getw()).replace("[", "").replace("]", "").replace(",", "")
        writefcn("%d %d %e %e %s\n" % (je, nfail, tcost, vcost, sw))
        
        # RProp update steps
        rpropmem = opt.rprop(G, rpropmem)
        steps = rpropmem[0]
        signs = rpropmem[1]
                        
        # update network weights
        w = la.vsum(net.getw(), la.vmul(steps, la.sax(-1, signs)))
        net.setw(w)
    
    # training complete             
    print "Training stopped"
    print "Final cost: ", mincost
    print "Final weights: ", wbest
        
    # return updated model
    net.setw(wbest)
    return net