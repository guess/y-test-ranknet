import math

ninp = 13;

def preproc(data):
    
    for q in data:
        for j in xrange(len(q)):
            
            # old data
            r = q[j]
            
            # new data
            p = [0] * (ninp + 2)
            
            # column 1: pass
            p[0] = r[0]
            
            # column 2: pass
            p[1] = r[1]
            
            # column 3: logarithm and center
            p[2] = math.log(r[2]) + 6.0
            
            # column 4: logarithm and center
            p[3] =  math.log(r[3]) + 6.3
            
            # column 5: logarithm and center 
            # additional column for missing values
            if r[4] == -1.0:
                p[4] = 0.0      # replace missing value
                p[5] = 1.0      # missing indicator
            else:
                p[4] = math.log(r[4]) + 5.6
                p[5] = 0.0      # missing indicator

            # column 6: pass
            p[6] = r[5]
            
            # column 7: logarithm
            p[7] = math.log(r[6])
            
            # column 8: 
            # truncate values above 1440*30 
            # transform scale to days
            if r[7] > 1440*30.0:
                p[8]  = 30.0
            else:
                p[8] = r[7] / 1440.0
            
            # column 9: logarithm and center 
            p[9] = math.log1p(r[8]) - 10.0
            
            # column 10:
            # values < 101 are missing values
            if r[9] <= 101.0:
                p[10] = 0.0
                p[11] = 1.0     # missing value indicator: value is missing
            else:
                p[10] = math.log(r[9]) - 9.0
                p[11] = 0.0     # missing value indicator: value is present
            
            # column 11: logarithm and center
            p[12] = math.log(r[10]) + 6.7
            
            # column 12: center
            p[13] = r[11] - 1.0
            
            # column 13:
            p[14] = math.log(r[12]) + 6.4
            
            # store updated data
            q[j] = p
            
            
            
            
            
            
            
            