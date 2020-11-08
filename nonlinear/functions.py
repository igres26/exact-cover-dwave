import numpy as np
from sympy import symbols, expand, Matrix, Identity
from sympy.matrices.dense import matrix2numpy
import collections


def num2list(n, b):
    """Turn a number into a list of its binary bits.
    Args:
        n (int): number to turn into binary.
        b (int): number of bits.
        
    Returns:
        x (list): list of binary bits that represent number n.
    """
    n = "{0:0{bits}b}".format(n, bits=b)
    x = []
    for i in range(b):
        x.append(int(n[i]))
    return x


def f_3(x1, x2, x3):
    """Function for the 3-7 nonlinear code.
    
    """
    f = np.array([x1*x2*x3 - x1*x2 - x1*x3 + x1 - x2*x3 + x2, 
                  x1*x2 - x1*x3 + x3, 
                  x1*x2*x3 - x1*x2 - 2*x2*x3 + x2 + x3,  
                  x1*x2*x3 - x1*x3 - x2*x3 + 1, 
                  x1*x2*x3 + x1*x3 - x1 - x2*x3 + 1, 
                  2*x1*x2*x3 - 2*x1*x2 + x1 - x2*x3 + x2, 
                  -2*x1*x2*x3 + x1*x3 + 2*x2*x3 - x2 - x3 + 1])
    return f


def f_8(x1, x2, x3, x4, x5, x6, x7, x8):
    """Function for the 8-16 nonlinear code.
    
    """
    f = np.array([-2*x1*x2 + x1 + x2,
                  -2*x2*x3 + x2 + x3,
                  -2*x3*x4 + x3 + x4,
                  -2*x4*x5 + x4 + x5,
                  2*x5*x6*x7*x8 - x5 - x6*x7*x8 + 1,
                  -8*x2*x4*x5*x6*x7*x8 + 8*x2*x4*x5*x6*x8 + 16*x2*x4*x5*x7*x8 - 8*x2*x4*x5*x7 - 8*x2*x4*x5*x8 + 4*x2*x4*x5 + 4*x2*x4*x6*x7*x8 - 4*x2*x4*x6*x8 - 8*x2*x4*x7*x8 + 4*x2*x4*x7 + 4*x2*x4*x8 - 2*x2*x4 + 4*x2*x5*x6*x7*x8 - 4*x2*x5*x6*x8 - 8*x2*x5*x7*x8 + 4*x2*x5*x7 + 4*x2*x5*x8 - 2*x2*x5 - 2*x2*x6*x7*x8 + 2*x2*x6*x8 +4*x2*x7*x8 - 2*x2*x7 - 2*x2*x8 + x2 + 4*x4*x5*x6*x7*x8 - 4*x4*x5*x6*x8 - 8*x4*x5*x7*x8 + 4*x4*x5*x7 + 4*x4*x5*x8 - 2*x4*x5 - 2*x4*x6*x7*x8 + 2*x4*x6*x8 +4*x4*x7*x8 - 2*x4*x7 - 2*x4*x8 + x4 - 2*x5*x6*x7*x8 + 2*x5*x6*x8 + 4*x5*x7*x8 - 2*x5*x7 - 2*x5*x8 + x5 + x6*x7*x8 - x6*x8 - 2*x7*x8 + x7 + x8,
                  8*x1*x2*x4*x6*x7*x8 - 8*x1*x2*x4*x6*x8 - 8*x1*x2*x4*x7*x8 + 8*x1*x2*x4*x7 + 8*x1*x2*x4*x8 - 4*x1*x2*x4 - 4*x1*x2*x6*x7*x8 + 4*x1*x2*x6*x8 + 4*x1*x2*x7*x8 - 4*x1*x2*x7 - 4*x1*x2*x8 + 2*x1*x2 - 4*x1*x4*x6*x7*x8 + 4*x1*x4*x6*x8 + 4*x1*x4*x7*x8 - 4*x1*x4*x7 - 4*x1*x4*x8 + 2*x1*x4 + 2*x1*x6*x7*x8 - 2*x1*x6*x8 -2*x1*x7*x8 + 2*x1*x7 + 2*x1*x8 - x1 - 4*x2*x4*x6*x7*x8 + 4*x2*x4*x6*x8 + 4*x2*x4*x7*x8 - 4*x2*x4*x7 - 4*x2*x4*x8 + 2*x2*x4 + 2*x2*x6*x7*x8 - 2*x2*x6*x8 -2*x2*x7*x8 + 2*x2*x7 + 2*x2*x8 - x2 + 2*x4*x6*x7*x8 - 2*x4*x6*x8 - 2*x4*x7*x8 + 2*x4*x7 + 2*x4*x8 - x4 - x6*x7*x8 + x6*x8 + x7*x8 - x7 - x8 + 1,
                  96*x1*x2*x3*x4*x5*x6*x7*x8 - 32*x1*x2*x3*x4*x5*x6*x7 - 32*x1*x2*x3*x4*x5*x6*x8 -32*x1*x2*x3*x4*x5*x7*x8 + 16*x1*x2*x3*x4*x5 - 48*x1*x2*x3*x4*x6*x7*x8 + 16*x1*x2*x3*x4*x6*x7 + 16*x1*x2*x3*x4*x6*x8 + 16*x1*x2*x3*x4*x7*x8 - 8*x1*x2*x3*x4 - 48*x1*x2*x3*x5*x6*x7*x8 + 16*x1*x2*x3*x5*x6*x7 + 16*x1*x2*x3*x5*x6*x8 + 16*x1*x2*x3*x5*x7*x8 - 8*x1*x2*x3*x5 + 24*x1*x2*x3*x6*x7*x8 - 8*x1*x2*x3*x6*x7 - 8*x1*x2*x3*x6*x8 - 8*x1*x2*x3*x7*x8 + 4*x1*x2*x3 - 48*x1*x2*x4*x5*x6*x7*x8 + 16*x1*x2*x4*x5*x6*x7 + 16*x1*x2*x4*x5*x6*x8 + 16*x1*x2*x4*x5*x7*x8 - 8*x1*x2*x4*x5 + 24*x1*x2*x4*x6*x7*x8 - 8*x1*x2*x4*x6*x7 - 8*x1*x2*x4*x6*x8 - 8*x1*x2*x4*x7*x8 + 4*x1*x2*x4 + 24*x1*x2*x5*x6*x7*x8 - 8*x1*x2*x5*x6*x7 - 8*x1*x2*x5*x6*x8 - 8*x1*x2*x5*x7*x8 + 4*x1*x2*x5 - 12*x1*x2*x6*x7*x8 + 4*x1*x2*x6*x7 + 4*x1*x2*x6*x8 + 4*x1*x2*x7*x8 - 2*x1*x2 - 48*x1*x3*x4*x5*x6*x7*x8 + 16*x1*x3*x4*x5*x6*x7 + 16*x1*x3*x4*x5*x6*x8 + 16*x1*x3*x4*x5*x7*x8 - 8*x1*x3*x4*x5 + 24*x1*x3*x4*x6*x7*x8 - 8*x1*x3*x4*x6*x7 - 8*x1*x3*x4*x6*x8 - 8*x1*x3*x4*x7*x8 + 4*x1*x3*x4 + 24*x1*x3*x5*x6*x7*x8 - 8*x1*x3*x5*x6*x7 - 8*x1*x3*x5*x6*x8 - 8*x1*x3*x5*x7*x8 + 4*x1*x3*x5 - 12*x1*x3*x6*x7*x8 + 4*x1*x3*x6*x7 + 4*x1*x3*x6*x8 + 4*x1*x3*x7*x8 - 2*x1*x3 + 24*x1*x4*x5*x6*x7*x8 -8*x1*x4*x5*x6*x7 - 8*x1*x4*x5*x6*x8 - 8*x1*x4*x5*x7*x8 + 4*x1*x4*x5 - 12*x1*x4*x6*x7*x8 + 4*x1*x4*x6*x7 + 4*x1*x4*x6*x8 + 4*x1*x4*x7*x8 - 2*x1*x4 - 12*x1*x5*x6*x7*x8 + 4*x1*x5*x6*x7 + 4*x1*x5*x6*x8 + 4*x1*x5*x7*x8 - 2*x1*x5 + 6*x1*x6*x7*x8 - 2*x1*x6*x7 - 2*x1*x6*x8 - 2*x1*x7*x8 + x1 - 48*x2*x3*x4*x5*x6*x7*x8 + 16*x2*x3*x4*x5*x6*x7 + 16*x2*x3*x4*x5*x6*x8 + 16*x2*x3*x4*x5*x7*x8 - 8*x2*x3*x4*x5 + 24*x2*x3*x4*x6*x7*x8 - 8*x2*x3*x4*x6*x7 -8*x2*x3*x4*x6*x8 - 8*x2*x3*x4*x7*x8 + 4*x2*x3*x4 + 24*x2*x3*x5*x6*x7*x8 - 8*x2*x3*x5*x6*x7 - 8*x2*x3*x5*x6*x8 - 8*x2*x3*x5*x7*x8 + 4*x2*x3*x5 - 12*x2*x3*x6*x7*x8 + 4*x2*x3*x6*x7 + 4*x2*x3*x6*x8 + 4*x2*x3*x7*x8 - 2*x2*x3 + 24*x2*x4*x5*x6*x7*x8 - 8*x2*x4*x5*x6*x7 - 8*x2*x4*x5*x6*x8 - 8*x2*x4*x5*x7*x8 + 4*x2*x4*x5 - 12*x2*x4*x6*x7*x8 + 4*x2*x4*x6*x7 + 4*x2*x4*x6*x8 + 4*x2*x4*x7*x8 -2*x2*x4 - 12*x2*x5*x6*x7*x8 + 4*x2*x5*x6*x7 + 4*x2*x5*x6*x8 + 4*x2*x5*x7*x8 - 2*x2*x5 + 6*x2*x6*x7*x8 - 2*x2*x6*x7 - 2*x2*x6*x8 - 2*x2*x7*x8 + x2 + 24*x3*x4*x5*x6*x7*x8 - 8*x3*x4*x5*x6*x7 - 8*x3*x4*x5*x6*x8 - 8*x3*x4*x5*x7*x8 + 4*x3*x4*x5 - 12*x3*x4*x6*x7*x8 + 4*x3*x4*x6*x7 + 4*x3*x4*x6*x8 + 4*x3*x4*x7*x8 -2*x3*x4 - 12*x3*x5*x6*x7*x8 + 4*x3*x5*x6*x7 + 4*x3*x5*x6*x8 + 4*x3*x5*x7*x8 - 2*x3*x5 + 6*x3*x6*x7*x8 - 2*x3*x6*x7 - 2*x3*x6*x8 - 2*x3*x7*x8 + x3 - 12*x4*x5*x6*x7*x8 + 4*x4*x5*x6*x7 + 4*x4*x5*x6*x8 + 4*x4*x5*x7*x8 - 2*x4*x5 + 6*x4*x6*x7*x8 - 2*x4*x6*x7 - 2*x4*x6*x8 - 2*x4*x7*x8 + x4 + 6*x5*x6*x7*x8 - 2*x5*x6*x7 - 2*x5*x6*x8 - 2*x5*x7*x8 + x5 - 3*x6*x7*x8 + x6*x7 + x6*x8 + x7*x8,
                  -8*x2*x5*x6*x7*x8 + 4*x2*x5*x6*x7 + 4*x2*x5*x8 - 2*x2*x5 + 4*x2*x6*x7*x8 - 2*x2*x6*x7 - 2*x2*x8 + x2 + 4*x5*x6*x7*x8 - 2*x5*x6*x7 - 2*x5*x8 + x5 - 2*x6*x7*x8 + x6*x7 + x8,
                  -2*x1*x6*x7*x8 + 4*x1*x6*x7 + 2*x1*x6*x8 - 2*x1*x6 + 2*x1*x7*x8 - 2*x1*x7 - 2*x1*x8 + x1 + x6*x7*x8 - 2*x6*x7 - x6*x8 + x6 - x7*x8 + x7 + x8,
                  -4*x1*x4*x6*x7 - 4*x1*x4*x6*x8 + 4*x1*x4*x6 + 4*x1*x4*x7 - 2*x1*x4 + 2*x1*x6*x7 + 2*x1*x6*x8 - 2*x1*x6 - 2*x1*x7 + x1 + 2*x4*x6*x7 + 2*x4*x6*x8 - 2*x4*x6 - 2*x4*x7 + x4 - x6*x7 - x6*x8 + x6 + x7,
                  8*x1*x3*x5*x6*x7*x8 - 8*x1*x3*x5*x6*x8 + 8*x1*x3*x5*x6 - 8*x1*x3*x5*x7*x8 + 8*x1*x3*x5*x8 - 4*x1*x3*x5 - 4*x1*x3*x6*x7*x8 + 4*x1*x3*x6*x8 - 4*x1*x3*x6 + 4*x1*x3*x7*x8 - 4*x1*x3*x8 + 2*x1*x3 - 4*x1*x5*x6*x7*x8 + 4*x1*x5*x6*x8 - 4*x1*x5*x6 + 4*x1*x5*x7*x8 - 4*x1*x5*x8 + 2*x1*x5 + 2*x1*x6*x7*x8 - 2*x1*x6*x8 +2*x1*x6 - 2*x1*x7*x8 + 2*x1*x8 - x1 - 4*x3*x5*x6*x7*x8 + 4*x3*x5*x6*x8 - 4*x3*x5*x6 + 4*x3*x5*x7*x8 - 4*x3*x5*x8 + 2*x3*x5 + 2*x3*x6*x7*x8 - 2*x3*x6*x8 +2*x3*x6 - 2*x3*x7*x8 + 2*x3*x8 - x3 + 2*x5*x6*x7*x8 - 2*x5*x6*x8 + 2*x5*x6 - 2*x5*x7*x8 + 2*x5*x8 - x5 - x6*x7*x8 + x6*x8 - x6 + x7*x8 - x8 + 1,
                  8*x2*x3*x4*x6*x7*x8 + 8*x2*x3*x4*x6*x8 - 8*x2*x3*x4*x6 - 8*x2*x3*x4*x7*x8 + 4*x2*x3*x4 - 4*x2*x3*x6*x7*x8 - 4*x2*x3*x6*x8 + 4*x2*x3*x6 + 4*x2*x3*x7*x8 - 2*x2*x3 - 4*x2*x4*x6*x7*x8 - 4*x2*x4*x6*x8 + 4*x2*x4*x6 + 4*x2*x4*x7*x8 - 2*x2*x4 + 2*x2*x6*x7*x8 + 2*x2*x6*x8 - 2*x2*x6 - 2*x2*x7*x8 + x2 - 4*x3*x4*x6*x7*x8 - 4*x3*x4*x6*x8 + 4*x3*x4*x6 + 4*x3*x4*x7*x8 - 2*x3*x4 + 2*x3*x6*x7*x8 + 2*x3*x6*x8 - 2*x3*x6 - 2*x3*x7*x8 + x3 + 2*x4*x6*x7*x8 + 2*x4*x6*x8 - 2*x4*x6 - 2*x4*x7*x8 + x4 - x6*x7*x8 - x6*x8 + x6 + x7*x8,
                  -16*x1*x3*x4*x5*x6*x8 + 16*x1*x3*x4*x5*x7*x8 - 16*x1*x3*x4*x5*x7 + 8*x1*x3*x4*x5+ 8*x1*x3*x4*x6*x8 - 8*x1*x3*x4*x7*x8 + 8*x1*x3*x4*x7 - 4*x1*x3*x4 + 8*x1*x3*x5*x6*x8 - 8*x1*x3*x5*x7*x8 + 8*x1*x3*x5*x7 - 4*x1*x3*x5 - 4*x1*x3*x6*x8+ 4*x1*x3*x7*x8 - 4*x1*x3*x7 + 2*x1*x3 + 8*x1*x4*x5*x6*x8 - 8*x1*x4*x5*x7*x8 + 8*x1*x4*x5*x7 - 4*x1*x4*x5 - 4*x1*x4*x6*x8 + 4*x1*x4*x7*x8 - 4*x1*x4*x7 + 2*x1*x4 - 4*x1*x5*x6*x8 + 4*x1*x5*x7*x8 - 4*x1*x5*x7 + 2*x1*x5 + 2*x1*x6*x8 - 2*x1*x7*x8 + 2*x1*x7 - x1 + 8*x3*x4*x5*x6*x8 - 8*x3*x4*x5*x7*x8 + 8*x3*x4*x5*x7 - 4*x3*x4*x5 - 4*x3*x4*x6*x8 + 4*x3*x4*x7*x8 - 4*x3*x4*x7 + 2*x3*x4 - 4*x3*x5*x6*x8 + 4*x3*x5*x7*x8 - 4*x3*x5*x7 + 2*x3*x5 + 2*x3*x6*x8 - 2*x3*x7*x8 +2*x3*x7 - x3 - 4*x4*x5*x6*x8 + 4*x4*x5*x7*x8 - 4*x4*x5*x7 + 2*x4*x5 + 2*x4*x6*x8- 2*x4*x7*x8 + 2*x4*x7 - x4 + 2*x5*x6*x8 - 2*x5*x7*x8 + 2*x5*x7 - x5 - x6*x8 + x7*x8 - x7 + 1,
                  -32*x1*x2*x3*x5*x6*x7*x8 + 32*x1*x2*x3*x5*x6*x8 - 16*x1*x2*x3*x5*x6 + 16*x1*x2*x3*x5*x7*x8 - 16*x1*x2*x3*x5*x8 + 8*x1*x2*x3*x5 + 16*x1*x2*x3*x6*x7*x8 - 16*x1*x2*x3*x6*x8 + 8*x1*x2*x3*x6 - 8*x1*x2*x3*x7*x8 + 8*x1*x2*x3*x8 - 4*x1*x2*x3 + 16*x1*x2*x5*x6*x7*x8 - 16*x1*x2*x5*x6*x8 + 8*x1*x2*x5*x6 - 8*x1*x2*x5*x7*x8 + 8*x1*x2*x5*x8 - 4*x1*x2*x5 - 8*x1*x2*x6*x7*x8 + 8*x1*x2*x6*x8 - 4*x1*x2*x6 + 4*x1*x2*x7*x8 - 4*x1*x2*x8 + 2*x1*x2 + 16*x1*x3*x5*x6*x7*x8 - 16*x1*x3*x5*x6*x8 + 8*x1*x3*x5*x6 - 8*x1*x3*x5*x7*x8 + 8*x1*x3*x5*x8 - 4*x1*x3*x5 - 8*x1*x3*x6*x7*x8 + 8*x1*x3*x6*x8 - 4*x1*x3*x6 + 4*x1*x3*x7*x8 - 4*x1*x3*x8 + 2*x1*x3 - 8*x1*x5*x6*x7*x8 + 8*x1*x5*x6*x8 - 4*x1*x5*x6 + 4*x1*x5*x7*x8 - 4*x1*x5*x8 + 2*x1*x5 + 4*x1*x6*x7*x8 - 4*x1*x6*x8 + 2*x1*x6 - 2*x1*x7*x8 + 2*x1*x8 - x1 + 16*x2*x3*x5*x6*x7*x8 - 16*x2*x3*x5*x6*x8 + 8*x2*x3*x5*x6 - 8*x2*x3*x5*x7*x8 + 8*x2*x3*x5*x8 - 4*x2*x3*x5 - 8*x2*x3*x6*x7*x8+ 8*x2*x3*x6*x8 - 4*x2*x3*x6 + 4*x2*x3*x7*x8 - 4*x2*x3*x8 + 2*x2*x3 - 8*x2*x5*x6*x7*x8 + 8*x2*x5*x6*x8 - 4*x2*x5*x6 + 4*x2*x5*x7*x8 - 4*x2*x5*x8 + 2*x2*x5 + 4*x2*x6*x7*x8 - 4*x2*x6*x8 + 2*x2*x6 - 2*x2*x7*x8 + 2*x2*x8 - x2 - 8*x3*x5*x6*x7*x8 + 8*x3*x5*x6*x8 - 4*x3*x5*x6 + 4*x3*x5*x7*x8 - 4*x3*x5*x8 + 2*x3*x5 + 4*x3*x6*x7*x8 - 4*x3*x6*x8 + 2*x3*x6 - 2*x3*x7*x8 + 2*x3*x8 - x3 + 4*x5*x6*x7*x8 - 4*x5*x6*x8 + 2*x5*x6 - 2*x5*x7*x8 + 2*x5*x8 - x5 - 2*x6*x7*x8 + 2*x6*x8 - x6 + x7*x8 - x8 + 1,
                  2*x3*x6*x7*x8 - 2*x3*x6*x7 + 2*x3*x6 - 2*x3*x7*x8 + 2*x3*x7 - x3 - x6*x7*x8 + x6*x7 - x6 + x7*x8 - x7 + 1])
    return f


def r_3():
    """Target state for the 3-7 nonlinear code.
    
    """
    return np.array([0, 1, 0, 1, 0, 0, 0])


def r_8():
    """Target state for the 8-16 nonlinear code.
    
    """
    return np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])


def to_ancilla(x1, x2, x3, a):
    """How a 3-body interaction between x1, x2 and x3 is decomposed using an ancilla a.
    
    ***NOT USED ANYMORE***
    
    """
    #return a*x2+x2*x3+2-2*a-x1-x2-x3
    return 2*x2*a - 2*x2 + 2*x3*a - 2*x3 - x1*a + x1 -3*a + 3 + x2*x3


def gadget(x, y, a):
    """Penalization function for the gadget substitution (x*y -> a).
    
    """
    return 3*a + x*y - 2*x*a - 2*y*a


def substitutions(sym):
    """Substitutions needed as 0 or 1 squared are equal to themselves.
    
    """
    s = []
    for x in sym:
        s.append((x**2, x))
    return s


def to_hamiltonian(f, r, sym):
    """Tranform the non-linear functions into a hamiltonian that minimizes the energy to find r and simplifies it.
    
    """
    h = np.sum((f-r)**2)
    h = expand(h)
    h = h.subs(substitutions(sym))
    return h


def to_two_body(h, ancillas, bits):
    """Turns 3-body hamiltonian into a 2-body one by use of an ancilla.
    
    ***SUBSTITUTED FOR GADGET TRANSFORMATION***
    
    """
    anc = 0
    for i in reversed(range(3, bits+1)):
        print()
        print(f'Dealing with {i}-term interactions.')
        terms_s, terms_c, overall_constant = symbolic_to_data(h)
        print(f'Total terms in the hamiltonian: {len(terms_s)}')
        for term in terms_s:
            if len(term) == i:
                xtemp1 = 1
                #for j in range(i-2):
                for j in range(int(i/3)):
                #for j in range(int(i/2)):
                    xtemp1 *= term[j]
                xtemp2 = 1
                #for j in range(i-2, i-1):
                for j in range(int(i/3), 2*int(i/3)):
                #for j in range(int(i/2), i-1):
                    xtemp2 *= term[j]
                xtemp3 = 1
                #for j in range(i-1, i):
                for j in range(2*int(i/3), i):
                #for j in range(i-1, i):
                    xtemp3 *= term[j]
                h = expand(h.subs(xtemp1*xtemp2*xtemp3, to_ancilla(xtemp1, xtemp2, xtemp3, ancillas[anc])))
                anc += 1
                print(f'Added ancilla number {anc}')
    print('Total ancillas used: {}\n'.format(anc))
    return h, anc


def separate_2_body(h, h_2):
    """Separate a hamiltonian between 1 and 2-order terms and the rest.
    
    """
    terms_s, terms_c, overall_constant = symbolic_to_data(h)
    for i in range(len(terms_s)):
        if len(terms_s[i]) == 2:
            h -= terms_c[i]*terms_s[i][0]*terms_s[i][1]
            h_2 += terms_c[i]*terms_s[i][0]*terms_s[i][1]
        elif len(terms_s[i]) == 1:
            h -= terms_c[i]*terms_s[i][0]
            h_2 += terms_c[i]*terms_s[i][0]
    return h, h_2
    

def add_gadget1(h, h_2, x1, x2, xa):
    """Add the gadget ancillas directly. No control concerns.
    
    """
    h, h_2 = separate_2_body(h, h_2)
    terms_s, terms_c, overall_constant = symbolic_to_data(h)
    h = expand(h.subs(x1*x2, xa))
    for j in range(len(terms_s)):
        if x1 in terms_s[j] and x2 in terms_s[j]:
            h += (1+np.abs(terms_c[j]))*gadget(x1, x2, xa)
    return h, h_2


def add_gadget2(h, h_2, x1, x2, xa):
    """Add the gadget ancillas trying to keep minimal control.
    
    """
    h, h_2 = separate_2_body(h, h_2)
    terms_s, terms_c, overall_constant = symbolic_to_data(h)
    h = expand(h.subs(x1*x2, xa))
    d_m = 0
    d_p = 0
    for j in range(len(terms_s)):
        if x1 in terms_s[j] and x2 in terms_s[j]:
            if terms_c[j] < 0:
                d_m += -terms_c[j]
            else:
                d_p += terms_c[j]
    h += (1+max(d_m, d_p))*gadget(x1, x2, xa)
    return h, h_2


def to_gadget_ruge1(h, x, ancillas, bits):
    h_2 = 0
    anc = 0
    h, h_2 = add_gadget1(h, h_2, x[0], x[1], ancillas[anc])
    anc += 1
    if len(ancillas) == 1:
        return h+h_2, anc
    h, h_2 = add_gadget1(h, h_2, x[2], x[3], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[0], x[2], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[1], x[3], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[0], x[3], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[1], x[2], ancillas[anc])
    anc += 1
    
    h, h_2 = add_gadget1(h, h_2, x[0+4], x[1+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[2+4], x[3+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[0+4], x[2+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[1+4], x[3+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[0+4], x[3+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[1+4], x[2+4], ancillas[anc])
    anc += 1
    
    h, h_2 = add_gadget1(h, h_2, x[0], x[9], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[1], x[9], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[2], x[8], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[3], x[8], ancillas[anc])
    anc += 1
    
    h, h_2 = add_gadget1(h, h_2, x[0+4], x[9+6], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[1+4], x[9+6], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[2+4], x[8+6], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[3+4], x[8+6], ancillas[anc])
    anc += 1
    
    h, h_2 = add_gadget1(h, h_2, x[8], x[9], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget1(h, h_2, x[8+6], x[9+6], ancillas[anc])
    anc += 1
    
    return h+h_2, anc


def to_gadget_ruge2(h, x, ancillas, bits):
    h_2 = 0
    anc = 0
    h, h_2 = add_gadget2(h, h_2, x[0], x[1], ancillas[anc])
    anc += 1
    if len(ancillas) == 1:
        return h+h_2, anc
    h, h_2 = add_gadget2(h, h_2, x[2], x[3], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[0], x[2], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[1], x[3], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[0], x[3], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[1], x[2], ancillas[anc])
    anc += 1
    
    h, h_2 = add_gadget2(h, h_2, x[0+4], x[1+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[2+4], x[3+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[0+4], x[2+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[1+4], x[3+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[0+4], x[3+4], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[1+4], x[2+4], ancillas[anc])
    anc += 1
    
    h, h_2 = add_gadget2(h, h_2, x[0], x[9], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[1], x[9], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[2], x[8], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[3], x[8], ancillas[anc])
    anc += 1
    
    h, h_2 = add_gadget2(h, h_2, x[0+4], x[9+6], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[1+4], x[9+6], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[2+4], x[8+6], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[3+4], x[8+6], ancillas[anc])
    anc += 1
    
    h, h_2 = add_gadget2(h, h_2, x[8], x[9], ancillas[anc])
    anc += 1
    h, h_2 = add_gadget2(h, h_2, x[8+6], x[9+6], ancillas[anc])
    anc += 1
    
    return h+h_2, anc


def check_energies(h, x, verbose=False):
    """Check all the possible energies and return the ground state.
    
    """
    ground = 10000
    for i in range(2**len(x)):
        val = num2list(i, len(x))
        s = []
        for j in range(len(x)):
            s.append((x[j], val[j]))
        en = h.subs(s)
        if verbose:
            print(f'For input: {val}    Energy output: {en}')
        if en < ground:
            ground = en
    if verbose:
        print()
    return ground   


def check_interactions(h, high = False):
    """Check the number of terms for all different order of interactions in a hamiltonian.
    
    - If high is True, print the symbols in the highest order of interactions.
    
    """
    terms_s, terms_c, overall_constant = symbolic_to_data(h)
    terms = {}
    for i in range(len(terms_s)):
        if len(terms_s[i]) not in terms.keys():
            terms[len(terms_s[i])] = 0
        terms[len(terms_s[i])] += 1
    if high:
        h = max(terms.keys())
        for i in range(len(terms_s)):
            if len(terms_s[i]) == h:
                print(terms_s[i])
        print()
    return terms


def check_two_body(h):
    """Check if given Hamiltonian only contains up to 2-body interactions.
    
    """
    terms_s, terms_c, overall_constant = symbolic_to_data(h)
    length = []
    for i in range(len(terms_s)):
        if len(terms_s[i]) > 2:
            raise ValueError(f'Error! Hamiltonian with more than two-body interactions. Found at least a {len(terms_s[i])}-body interaction.')

    
def symbolic_to_data(symbolic_hamiltonian):
    """Transforms a symbolic Hamiltonian to lists of every term.
        
    Args:
        symbolic_hamiltonian: The full Hamiltonian written with symbols.
    
    Returns:
        matching list of the symbols in each term of the hamiltonian and the corresponding constant.
    """ 
    terms_s = []
    terms_c = []
    overall_constant = 0
    for term in symbolic_hamiltonian.args:
        if not term.args:
            expression = (term,)
        else:
            expression = term.args

        symbols = [x for x in expression if x.is_symbol]
        numbers = [x for x in expression if not x.is_symbol]

        if len(numbers) > 1:
            raise ValueError("Hamiltonian must be expanded before using this method.")
        elif numbers:
            constant = float(numbers[0])
        else:
            constant = 1

        if not symbols:
            overall_constant += constant
            
        terms_s.append(symbols)
        terms_c.append(constant)
    
    return terms_s, terms_c, overall_constant
    
    
def symbolic_to_dwave(symbolic_hamiltonian, symbol_num):
    """Transforms a symbolic Hamiltonian to a dictionary of targets and matrices.
    
    Works for Hamiltonians with one and two qubit terms only.
    
    Args:
        symbolic_hamiltonian: The full Hamiltonian written with symbols.
        symbol_num: Dictionary that maps each symbol that appears in the 
            Hamiltonian to its target.
    
    Returns:
       Q (dict): Dictionary with the interactions to send to the DWAVE machine.
       overall_constant (int): Constant that cannot be given to DWAVE machine.
    """ 
    Q = {}
    overall_constant = 0
    for term in symbolic_hamiltonian.args:
        if not term.args:
            expression = (term,)
        else:
            expression = term.args

        symbols = [x for x in expression if x.is_symbol]
        numbers = [x for x in expression if not x.is_symbol]

        if len(numbers) > 1:
            raise ValueError("Hamiltonian must be expanded before using this method.")
        elif numbers:
            constant = float(numbers[0])
        else:
            constant = 1

        if not symbols:
            overall_constant += constant

        elif len(symbols) == 1: 
            target = symbol_num[symbols[0]]
            #print(symbols[0], target)
            Q[(target, target)] = constant

        elif len(symbols) == 2:
            target1 = symbol_num[symbols[0]]
            target2 = symbol_num[symbols[1]]
            #print(symbols[0], target1)
            #print(symbols[1], target2)
            Q[(target1, target2)] = constant

        else:
            raise ValueError("Only one and two qubit terms are allowed.")
    
    return Q, overall_constant