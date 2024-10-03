import numpy as np
import shapely
import pqdict 


np.set_printoptions(precision=3, suppress = True)

def meters2cells(x, m, r):
    """
    x: Coords of the origin
    m: minimum continuous metric coordinates contained in the grid
    r: grid resolution
    
    return:
        the coords of the point in cell
    """
    return np.floor((x-m)/r)

def cells2meters(x, m, r):
    """ 
    return:
        The coords of the point in meterss
    """
    return (x+0.5)*r+m

def check_collision():
    return None

def main():
 
   return None
 
if __name__ == '__main__':
   main()