import numpy as np
import shapely
import pqdict 
from shapely import box, LineString, intersection, Polygon

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
    x: Coords of the origin
    m: minimum continuous metric coordinates contained in the grid
    r: grid resolution
    
    return:
        The coords of the point in meterss
    """
    return (x+0.5)*r+m


def check_collision(curr,next,blocks):
    """
    Check the collision of the given line segment (curr to next) and the blocks 

    The main idea of this function is project the line segment and blocks to x,y,z plain
    so we can get (trajx,trajy,trajz) and (ix,iy,iz).
    Then we check if there is intersection on every plain.
    If all of the plain have intersection, that means there is intersection in 3D world
    If intersection only occurs on not more than 2 plain, there is not
    
    Args:
        curr (x,y,z): current node
        next (x,y,z): next node
        blocks (n*9 ndarray): all the blocks

    Returns:
        collision(bool): if there is collision
    """
    collision = False
    
    trajx = LineString([(curr[1],curr[2]),(next[1],next[2])]) # projection of the line on x plane
    trajy = LineString([(curr[0],curr[2]),(next[0],next[2])]) # projection of the line on y plane  
    trajz = LineString([(curr[0],curr[1]),(next[0],next[1])]) # projection of the line on z plane
    for i in blocks: # (xmin,ymin,zmin,xmax,ymax,zmax,rbg,rbg,rbg)
        xmin,ymin,zmin,xmax,ymax,zmax = i[:6]
        ix = box(ymin,zmin,ymax,zmax)
        iy = box(xmin,zmin,xmax,zmax)
        iz = box(xmin,ymin,xmax,ymax)
        
        # the function intersection returns a line segment
        # If the len of that line segment is 0 it means the line doesn't exist
        # which means there is no intersection
    
        x_intersect = intersection(trajx,ix).length
        y_intersect = intersection(trajy,iy).length
        z_intersect = intersection(trajz,iz).length

        if trajx.length == 0:
            x_intersect = 1
        if trajy.length == 0:
            y_intersect = 1
        if trajz.length == 0:
            z_intersect = 1
        
        if (x_intersect != 0) and\
        (y_intersect != 0) and\
        (z_intersect != 0): 
            collision = True
        
    return collision


def check_collision3d(curr,next,blocks):
    """
    Check the collision of the given line segment (curr to next) and the blocks in 3d cases
    
    but is was not working that well

    Args:
        curr (x,y,z): current node
        next (x,y,z): next node
        blocks (n*9 ndarray): all the blocks

    Returns:
        collision(bool): if there is collision
    """
    collision = False
    
    traj = LineString([curr,next]) # projection of the line on x plane
    for i in blocks: # (xmin,ymin,zmin,xmax,ymax,zmax,rbg,rbg,rbg)
        xmin,ymin,zmin,xmax,ymax,zmax = i[:6]
        nodes = np.array([
            [xmin,ymin,zmin],
            [xmin,ymin,zmax],
            [xmin,ymax,zmax],
            [xmin,ymax,zmin],
            [xmax,ymax,zmin],
            [xmax,ymax,zmax],
            [xmax,ymin,zmax],
            [xmax,ymin,zmin],
            [xmin,ymin,zmin]
        ])

        if len(intersection(Polygon(nodes),traj).coords) != 0:
            collision = True
        
    return collision






def check_collision_for_path(path,blocks):
    """
    check if there is collision after we find a path
    
    for part1 specifically

    Args:
        path (n*3 ndarray): a list of nodes, which represent the path
        blocks (n*9): _description_
    """
    there_is_collision = False
    for i in range(path.shape[0] - 1):
        seg_i = check_collision(path[i],path[i+1],blocks)
        if seg_i:
            there_is_collision = True
        # print(seg_i)
        
    return there_is_collision
