import numpy as np
import shapely
from shapely import box, LineString, normalize, Polygon,intersection,intersection_all
import math
from pqdict import pqdict
import heapq

# traj = np.array([
#           [2.3  ,2.3,  1.3  ],
#           [2.589,2.589,1.589],
#           [2.877,2.877,1.877],
#           [3.166,3.166,2.166],
#           [3.455,3.455,2.455],
#           [3.743,3.743,2.743],
#           [4.032,4.032,3.032],
#           [4.321,4.321,3.321],
#           [4.609,4.609,3.609],
#           [4.898,4.898,3.898],
#           [5.187,5.187,4.187],
#           [5.475,5.475,4.475],
#           [5.764,5.764,4.764],
#           [6.053,6.053,5.053],
#           [6.341,6.341,5.341],
#           [6.695,6.695,5.341],
#           [6.984,6.984,5.63 ]
#           ])

# bounday = np.array([[-5,-5,-5,10,10,10,120,120,120]])
# blocks = np.array([[4.5,4.5,2.5,5.5,5.5,3.5,120,120,120]])

# def check_collision(curr,next,blocks):
#   """ 
#   (x,y,z)
#   """
#   collision = False
  
#   trajx = LineString([(curr[1],curr[2]),(next[1],next[2])]) # projection of the line on x plane
#   trajy = LineString([(curr[0],curr[2]),(next[0],next[2])]) # projection of the line on y plane  
#   trajz = LineString([(curr[0],curr[1]),(next[0],next[1])]) # projection of the line on z plane
#   for i in blocks: # (xmin,ymin,zmin,xmax,ymax,zmax,rbg,rbg,rbg)
#     xmin,ymin,zmin,xmax,ymax,zmax = i[:6]
#     ix = box(ymin,zmin,ymax,zmax)
#     iy = box(xmin,zmin,xmax,zmax)
#     iz = box(xmin,ymin,ymax,xmax)
    
#     x_intersect = len(intersection(trajx,ix).coords)
#     y_intersect = len(intersection(trajy,iy).coords)
#     z_intersect = len(intersection(trajz,iz).coords)
    
#     # print(x_intersect,y_intersect,z_intersect)
#     if (x_intersect != 0) and\
#        (y_intersect != 0) and\
#        (z_intersect != 0): # 
#          collision = True
    
#   return collision


# for i in traj:
#   there_is_collision = False
#   seg_i_collision = check_collision(i,i+1,blocks)
#   if seg_i_collision:
#     there_is_collision = True
#   print(there_is_collision)

# if there_is_collision:
#   print("there is collision")
# else:
#   print("the trajctory is collision free")


  



# line = LineString([(1.,1.),(2.,2.)])
# poly = Polygon([(9.,9.),(9.,10.),(10.,10.),(10.,9.),(9.,9.)])
# inter = intersection(line,poly)
# print(len(line.coords))
# print(len(inter.coords))
# print(type(inter))




import utils

block = np.array([[1.,0.,0.,1.1,19.,5.]])
curr = np.array([0.75,1.25,4.75])
next = np.array([1.25,1.25,4.75])

check = utils.check_collision3d(curr,next,block)
print(check)






# class AStarNode(object):
#   def __init__(self, pqkey, coord, hval):
#     self.pqkey = pqkey
#     self.coord = coord
#     self.g = math.inf
#     self.h = hval
#     self.parent_node = None
#     self.parent_action = None
#     self.closed = False
#   def __lt__(self, other):
#     return self.g < other.g 

# # for every node: [x,y,z,hval]
# nodes = np.array([
#     [1,1,1,10],[2,1,1,8],
#     [3,2,1,20],[4,1,2,41],
#     [5,2,1,32],[6,2,2,6],
#     [7,1,2,15],[8,2,2,18],
#     [9,1,1,22],[10,3,1,7],
#     [11,1,3,30],[12,3,1,2],  
# ])

# pq_nodes = pqdict({},key=lambda x:x.h)

# for i in nodes:
#     coords,hval = i[:3],i[-1]
#     # name = "{},{},{}".format(i[0],i[1],i[2])
#     name = tuple(coords)
#     # creating astar node object
#     a = AStarNode(name,coords,hval)
#     # add it into pqdict
#     pq_nodes[name] = a


# # print(pq_nodes.keys)
# print(pq_nodes.values)


# while pq_nodes !=  None:
#   # print(list(pq_nodes.keys()))   
#   node = pq_nodes.popitem()
#   print(node[1].h)
#   if len(pq_nodes) == 0:
#     print("empty")
#     break






 







