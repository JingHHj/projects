# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot


class SamplingBased:
    def __init__(self,boundary,blocks,start,goal):
        if len(boundary.shape) > 1:
            boundary = np.copy(boundary[0,:])
        self.boundary = np.array([
            (boundary[0],boundary[3]),
            (boundary[1],boundary[4]),
            (boundary[2],boundary[5])
        ])
        self.blocks = blocks[:,:6]
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.SearchSpace = SearchSpace(self.boundary, self.blocks[:6])
        # create Search Space
        self.path = None

  
    def rrt_plan(self):
        q = 8  # length of tree edges
        r = 1  # length of smallest edge to check for intersection with obstacles
        max_samples = 1024  # max number of samples to take before timing out
        prc = 0.1  # probability of checking for a connection to goal
        

        # create rrt_search
        self.algorithm = RRT(self.SearchSpace, q, self.start, self.goal, max_samples, r, prc)
        self.path = self.algorithm.rrt_search()
        return None

    def visiulization(self):
        # plot
        plot = Plot("rrt_3d")
        plot.plot_tree(self.SearchSpace,self.algorithm.trees)
        if self.path is not None:
            plot.plot_path(self.SearchSpace, self.path)
        plot.plot_obstacles(self.SearchSpace, self.blocks)
        plot.plot_start(self.SearchSpace, self.goal)
        plot.plot_goal(self.SearchSpace, self.goal)
        plot.draw(auto_open=True)
    

# def nothing():
#     return None 
# def main():
    
#     # X_dimensions = np.array([(0, 100), (0, 100), (0, 100)])  # dimensions of Search Space
#     X_dimensions = np.array([0,0,0,100,100,100]) 
#     # obstacles
#     Obstacles = np.array(
#         [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
#         (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])
#     x_init = (0, 0, 0)  # starting location
#     x_goal = (100, 100, 100)  # goal location
#     rrt = SamplingBased(X_dimensions,Obstacles,x_init,x_goal)
#     rrt.rrt_plan()
#     # rrt.visiulization()
    
#     return None
 
# if __name__ == '__main__':
#     nothing()
# #    main()