"""
RankNet

See: 
[1] C
[2] C. Burges, From RankNet to LambdaRank to LambdaMART: An Overview

RankNet uses a pair-wise cost function to achieve learning
of relevance relationships between items of the same query. 
The RankNet cost function may be used on top of any learning algorithm.
Following [1] we use a standard feed-forward neural network. 

"""

import math, la

sgm = 1.0

def s(x, y):
    if x > y: return 1.0
    elif x == y: return 0.0
    else: return -1.0

def cost(query, model):
            
    # Compute predictions
    scores = [];
    
    for u in query:
        scores += model.apply(u[2:])
            
    # Sum contributions from all pairs
    C = 0.0
        
    for j in xrange(0, len(query)-1):
        for k in xrange(j+1, len(query)):
            if query[j][1] != query[k][1]:
                a = 1.0 - s(query[j][1], query[k][1])
                d = scores[j] - scores[k]
                C += a * d * sgm / 2.0 + math.log1p( math.exp( - sgm * d ) )

    return C
    
def gradient(query, model):  
    
    # Compute predictions and gradients for each entry in query
    scores = [];
    grads = [];
    
    for u in query:
        scores += model.apply(u[2:])         # features are in rows from 
        grads  += [ model.backprop([1]) ]        # assuming model has only one output
         
    # Index elements for lambda computations
    bots = [j for j in xrange(0, len(query)) if query[j][1] == 0]  # index of elements with label 0
    tops = [j for j in xrange(0, len(query)) if query[j][1] == 1]  # index of elements with label 1
        
    # Compute lambdas
    lambs = [0] * len(query)
    
    for j in xrange(0, len(query)):
        if query[j][1] == 1:                # top sample
            # sum over bottom elements
            for k in bots:
                lambs[j] += lmb(scores[j], scores[k])
        else:                               # bottom sample
            # sum over top elements
            for k in tops:
                lambs[j] -= lmb(scores[k], scores[j])
                
    # Final gradient is a weighted sum of individual gradients
    return la.gax(grads, lambs)
                
def lmb(x, y):
    return  - sgm / ( 1.0 + math.exp( sgm * (x - y) ))
    