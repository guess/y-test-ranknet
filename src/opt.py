def rprop(grad, mem):       
    """
    One iteration of RProp optimization algorithm. Adapts variable
    step values using only signs of gradient on the current and the previous iterations.   
    """

    # standard rprop options
    incr = 1.1      # step increment factor
    decr = 0.5      # step decrement factor
    
    rmax = 50       # maximal value of step
    rmin = 1.e-8    # minimal value of step
    
    # working memory
    steps = mem[0]          # current step sizes
    sgn_prev = mem[1]       # gradient signs from previous call
    
    # signs of current gradient
    s = [ 1 if g >= 0 else -1 for g in grad ]
    
    # for each gradient component
    for j in xrange(len(s)):
        # if the same sign    
        if s[j] * sgn_prev[j] >= 0:
            steps[j] = incr * steps[j]          # increase step
            if steps[j] > rmax:                 # limit step size
                steps[j] = rmax
        # if different sign
        else:
            steps[j] = decr * steps[j]          # decrease step      
            if steps[j] < rmin:                 # limit step size
                steps[j] = rmin
    
    # return updated working memory         
    return (steps, s)       
    
