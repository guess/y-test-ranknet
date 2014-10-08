"""
RankNet

See: 
[1] C. Burges et al., Learning to Rank using Gradient Descent 
[2] C. Burges, From RankNet to LambdaRank to LambdaMART: An Overview

"""

import math, la
     
def cost(query, model, sigma):
            
    # Compute model outputs for each sample in query
    scores = [ model.apply(u[2:])[0] for u in query ]
            
    # Find all pairs
    pairs = []
    C = 0
    for i in xrange(len(query)):
        for j in xrange(i+1, len(query)):
            if query[i][1] != query[j][1]:
                s = S( query[i][1], query[j][1] )                       # 
                delta = scores[i] - scores[j]                           # scores difference
                prob = ( 1.0 + math.tanh(sigma*delta/2.0) ) / 2.0       # estimated probability of ranking i > j     
                c = (1.0 - s) * sigma * delta / 2.0 - math.log(prob)    # cost contribution
                pairs.append((i, j, prob, c))                                
                C += c                                                  # total cost
    
    return (C, scores, pairs)
    
        
def gradient(query, model, sigma):  
    
    # Compute model outputs for each sample in query
    scores = []
    grads = []
    
    for u in query:
        scores += model.apply(u[2:])
        grads += [ model.backprop([1]) ]
    
    # Find all pairs, their S_ij and lambda_ij
    pairs = []
    for i in xrange(len(query)):
        for j in xrange(i+1, len(query)):
            if query[i][1] != query[j][1]:
                s = S( query[i][1], query[j][1] )
                lambd = lmb( scores[i], scores[j], s, sigma )
                pairs.append((i, j, lambd))
    
    # Sum lambda's by sample
    lambds = [0] * len(query)
    for t in pairs:
        lambds[ t[0] ] += t[2]
        lambds[ t[1] ] -= t[2]
                
    # Final gradient is a weighted sum of individual gradients
    return la.gax(grads, lambds)
                    
def lmb(x, y, S, sigma):
    """
    Lambda coefficient for a pair of samples. [2, equation (3)]
    """
    a = (1.0 - S) / 2.0
    b = ( 1.0 + math.tanh(sigma*(y - x)/2.0) ) / 2.0
    return ( a - b ) * sigma  

def S(x, y):
    if x > y: return 1.0
    elif x == y: return 0.0
    else: return -1.0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    