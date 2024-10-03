from time import time
import numpy as np
import utils 
from utils import sim_time, time_step

class Environment:
    """
    Aim: initialize the environment with obstacles and reference trajectory
    """
    def __init__(self,iter_step):
        self.obs = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
        self.iter_step = iter_step

        # initialize reference trajectory
        ref_traj = []
        for i in range(self.iter_step):
            ref_traj.append(utils.lissajous(i))
        self.ref_traj = np.asarray(ref_traj)

def test(): 
    """
    Aim: just using for testing the initialization of the environment
    """
    iter_step = int(np.floor(sim_time / time_step))
    env = Environment(iter_step)
    print(env.ref_traj)
    return env.ref_traj

if __name__ == "__main__":
    test()
        

