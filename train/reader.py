def read(fname, jstartline = 0, maxlines = 0):
    """
    Read Y data files
    """
    
    f = open(fname, "r")     # open data file
        
    count = 0       # line count
    jq = -1      # current jq id
    data = []       # list of data, partitioned by jq
    
    # read file line by line
    for line in f:
        
        # skip until requested line
        if count < jstartline: 
            count += 1
            continue
        
        # break line into tokens
        elements = line.split()
        
        # column 0: query number
        q = int(elements[0])
        
        if q > jq:
            # new partition
            jq = q
            query = []
            data.append(query)
            
        # column 1: relevance label    
        r = int(elements[1])
        
        # columns 2 to 12: features
        x = map(float, elements[2:])
        
        # add recored to partition
        query.append([q, r] + x)
        
        # check for stopping 
        count += 1
        if maxlines > 0 and count >= jstartline + maxlines:
            break
    
    # done    
    return data       
