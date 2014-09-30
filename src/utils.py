

class Matrix(list):
    """
    Simple matrix implementation using lists
    """

    
    
def zeros(n, m):
    """
    Create matrix filled with zeros
    """
    
    # create new matrix object
    matrix = Matrix()
    # fill with zeros
    for _ in xrange(0, n):
        matrix.append([0] * m)
    # done
    return matrix

def ones(n, m):
    """
    Create matrix filled with ones
    """
    
    # create new matrix object
    matrix = Matrix()
    # fill with ones
    for _ in xrange(0, n):
        matrix.append([1] * m)
    # done
    return matrix

def eye(n):
    """
    Create identity matrix 
    """
    
    # create new matrix object
    matrix = Matrix()
    # fill diagonal with ones
    for j in range(0, n):
        matrix.append([0] * n)
        matrix[j][j] = 1
    # done
    return matrix























