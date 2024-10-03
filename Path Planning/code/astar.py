# priority queue for OPEN list
from pqdict import pqdict
import numpy as np
import math
from utils import check_collision
import utils

class AStarNode(object):
  def __init__(self, pqkey, coord, hval):
    self.pqkey = pqkey # the key in the OPEN 
    self.coord = coord # coordinate
    self.g = math.inf # estimate cost
    self.h = hval # heuristic
    self.f = self.g + self.h 
    self.parent_node = None
    
  def get_f(self):
    """ 
    renew f
    """
    self.f = self.g + self.h 
  
     
class AStar(object):
  @staticmethod
  def plan(env,epsilon = 1):
    # Initialize the graph and open list
    CLOSED = {}
    OPEN = pqdict({},key=lambda x:x.f)

    # initiate start node
    start = AStarNode(tuple(env.start), env.start, env.getHeuristic(env.start))
    start.g = 0
    start.get_f
    OPEN.additem(tuple(env.start),start)
    # j = 0
    while len(OPEN) != 0:

      # if j <= 5000:
      #   if j%100 == 0:
      #     print(j)
      # else:
      #   if j%500 == 0:
      #     print(j)
      # j = j + 1
      
      # pop the item with the lowest f 
      curr = OPEN.popvalue()
      # find its neighbours
      neighbours = env.find_neighbours(curr.coord)
      curr_meter = utils.cells2meters(curr.coord,env.base,env.res)
      for i in neighbours:
        i_meter = utils.cells2meters(i,env.base,env.res)
        if env.if_out(i): 
            # check if the next node is out of map or inside of blocks
          continue
        elif tuple(i) in CLOSED: 
            # check if the next node is already in CLOSED
          continue
        elif check_collision(curr_meter,i_meter,env.blocks):
            # check if there is collision from current node to the next
          continue
        else:
          pqkey = tuple(i)

          if pqkey in OPEN:
            # if the next node is already in OPEN
            next = OPEN[pqkey]
          else:
            next = AStarNode(pqkey,i,env.getHeuristic(i))
          
          cij = math.sqrt(np.sum((curr.coord - next.coord)**2))
          # gi+ cij < gj
          if curr.g + cij < next.g:
            next.g = curr.g + cij # relabeling
            next.parent_node = curr # keep on track of the parent node
            next.get_f()
            OPEN[pqkey] = next
          else:
            continue
      CLOSED[tuple(curr.coord)] = curr
      
      if tuple(env.goal) in CLOSED:
        break
    
    # trace back to find the trajectory
    traj = np.copy(env.goal).reshape(1,3)
    while True: 
      parent = CLOSED[tuple(traj[0])]
      traj = np.vstack((parent.parent_node.coord,traj))
      if np.all(traj[0] == env.start):
        break

    # print("****")
    # print(env.start)
    # print(traj)
    # print(env.goal)
    # print("****")
    return traj


class Environment(object):
  def __init__(self,blocks,boundary,start,goal,res):
    self.res = res # the resolution of the cell
    self.base = np.copy(boundary[0,:3])
    self.start = utils.meters2cells(goal,self.base,self.res) # discretization
    self.goal = utils.meters2cells(start,self.base,self.res) # discretization
        # xmin ymin zmin xmax ymax zmax r g b (many)
    self.boundary = boundary
        # xmin ymin zmin xmax ymax zmax
    self.blocks = blocks    
                  
  def find_neighbours(self,node):
    """ 
    Helping the input node to find its neighbour cells
    """
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1)
    dR = dR.T
    neighbours = np.tile(node,(26,1))
    return neighbours + dR
  
  def getHeuristic(self,curr):
    """
    Get the heuristic of current node
    using manhattan distance (1-norm)

    Args:
        curr: currrent node (x,y,z)
    """
    return math.sqrt(np.sum((curr - self.goal)**2))
  
  def if_out(self,node):
    """
    Tell if the given node is out of boundary or inside of obstacles

    node:
        out(bool): 
    """
    out = False     
    x,y,z = utils.cells2meters(node,self.base,self.res)
    if( x < self.boundary[0,0] or x > self.boundary[0,3] or \
        y < self.boundary[0,1] or y > self.boundary[0,4] or \
        z < self.boundary[0,2] or z > self.boundary[0,5] ):
        out = True
    else: 
      for k in range(self.blocks.shape[0]):
          if( x >= self.blocks[k,0] and x <= self.blocks[k,3] and\
              y >= self.blocks[k,1] and y <= self.blocks[k,4] and\
              z >= self.blocks[k,2] and z <= self.blocks[k,5] ):
            out = True
            break
    return out
    
    

