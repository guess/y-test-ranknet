import la
from sys import float_info

def qorder(data):
    # Check that queries are ordered and all ids are present
    
    query = -1
    
    for q in data:
        query += 1
        assert(query == q[0][0])
    
def stats(data):
    # See: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm
    
    nx = len(data[0][0])        # number of variables
    
    mins = [float_info.max] * nx   
    maxs = [-float_info.max] * nx
    means = [0] * nx            # means of variables
    variances = [0] * nx              # variances of variables
   
    # sample counter
    j = 0
 
    # run through data once
    for q in data:
        for x in q:
            # update counter        
            j = j + 1
            
            # update mins and maxs
            for k in xrange(len(x)):
                if x[k] > maxs[k]: maxs[k] = x[k]
                if x[k] < mins[k]: mins[k] = x[k]
            
            # update means and variances
            delta = la.vsum(x, la.sax(-1.0, means))
            means = la.vsum(means, la.sax(1.0/j, delta))
            variances = la.vsum(variances, la.vmul(delta, la.vsum(x, la.sax(-1.0, means))))
 
    # normalize variance
    variances = la.sax( 1.0 / (j - 1), variances )
 
    return [mins, maxs, means, variances]













