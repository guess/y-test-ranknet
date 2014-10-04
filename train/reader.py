

def read(fname, maxlines = 0):
    
    f = open(fname, "r")     # open data file
        
    count = 0       # line count
    query = -1      # current query id
    data = []       # list of data, partitioned by query
    
    # read file line by line
    for line in f:
        
        # break line into tokens
        elements = line.split()
        
        # column 0: query number
        q = int(elements[0])
        
        if q > query:
            # new partition
            query += 1
            data.append([])
            
        # column 1: relevance label    
        r = int(elements[1])
        
        # columns 2 to 12: features
        x = map(float, elements[2:])
        
        # add recored to partition
        data[query].append([q, r] + x)
        
        # check for stopping 
        count += 1
        if maxlines > 0 and count >= maxlines:
            break
    
    # done    
    return data       
