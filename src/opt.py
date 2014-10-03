"""
Numerical optimization 
======================

"""

import la

def descent(x, gx, a):
    dx = - la.inner(a, gx)
    return dx

