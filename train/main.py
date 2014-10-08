import ffnet, activation, trainsg, reader, la, preprocess, ranknet
import random


def main():

    # load training data
    data = reader.read("../data/log_1_fixed.txt")
    
    # preprocess 
    preprocess.preproc(data)
    
    # shuffle data
    perm = range(len(data))
    random.shuffle(perm)
    
    # train a network
    ntrain = 100000
    dtrain = la.idxview(data, perm[:ntrain])
    
    net = train(0, dtrain)
    
    # evaluate on training data
    evaluate(dtrain, net)
        
    # evaluate on test data
    devaluate = la.idxview(data, perm[ntrain:2*ntrain])
    evaluate(devaluate, net)
    

def train(k, data):
    
    # initialize new model
    layers = [13, 8, 1]
    activf = [activation.linear(), activation.tanh(), activation.sigmoid()]  
    net = ffnet.FFNet(layers, activf)
    net.initw(0.1) 
    
    # use default training options
    opts = trainsg.options()
    opts.rate = 2.e-4
    
    # write function
    f = open("../output/train-%s.txt" % k, "w+")
    writefcn = lambda s: f.write(s)

    # training
    net = trainsg.train(data, opts, net, writefcn)

    # close file
    f.close()
    
    # return trained network
    return net

def evaluate(data, net):
    
    margins = [0.01*j for j in xrange(51)]
    lossmatr = [[[0 for _ in xrange(3)] for _ in xrange(3)] for _ in margins] 
    
    for q in data:
        
        C = ranknet.cost(q, net, 2.0, True)
        labels = C[1]
        pairs = C[3]
        for pair in pairs:
            
            i = pair[0]
            j = pair[1]
            p = pair[2]
            
            s = 1
            if labels[i] > labels[j]: s = 2
            if labels[i] < labels[j]: s = 0
            
            for jm in xrange(len(margins)):

                z = 1
                if p > 0.5 + margins[jm]: z = 2
                if p < 0.5 - margins[jm]: z = 0
                                
                lossmatr[jm][z][s] += 1
        
    for jm in xrange(len(margins)):
        print margins[jm], lossmatr[jm]
    


if __name__ == '__main__':
    main()