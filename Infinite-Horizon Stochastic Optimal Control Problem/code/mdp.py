import numpy as np
from dataclasses import dataclass, field
from environment import Environment
import utils
from joblib import Parallel, delayed, Memory
from numba import jit
import pickle
import gc
dir = "*/cache_dir"
new = np.newaxis


"""
run the init and init2 function to initialize the MDP object
run the vi function to do the value iteration
run the findTraj function to find the trajectory
"""


@dataclass
class MDP:
    nt: int # number of time steps
    nx: int
    ny: int
    nth: int
    nv: int
    nw: int
    ref_traj: np.ndarray  # (nt,3)
    nneighbors: int = 6  # no diagnal movement

    def __post_init__(self):
        self.x_res = (3-(-3))/self.nx
        self.y_res = (3-(-3))/self.ny
        self.th_res = (np.pi-(-np.pi))/self.nth
        self.v_res = (utils.v_max - utils.v_min)/self.nv
        self.w_res = (utils.w_max - utils.w_min)/self.nw

        # transition matrix: 100*10*10*10*5*10*6*4
        self.P = np.zeros((self.nt,self.nx,self.ny,self.nth,self.nv,self.nw,self.nneighbors,4)) 
        self.l = np.zeros((self.nt,self.nx,self.ny,self.nth,self.nv,self.nw,self.nneighbors)) # stage 
       
      
        xidx = np.arange(self.nx)[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
        yidx = np.arange(self.ny)[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        thidx = np.arange(self.nth)[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
        vidx = np.arange(self.nv)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
        widx = np.arange(self.nw)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]


        xidx = np.tile(xidx,(1,self.ny,self.nth,self.nv,self.nw))
        yidx = np.tile(yidx,(self.nx,1,self.nth,self.nv,self.nw))
        thidx = np.tile(thidx,(self.nx,self.ny,1,self.nv,self.nw))
        vidx = np.tile(vidx,(self.nx,self.ny,self.nth,1,self.nw))
        widx = np.tile(widx,(self.nx,self.ny,self.nth,self.nv,1))
        
        # righ now is still the self.current state
        # in cell 
        self.current = np.stack((xidx,yidx,thidx,vidx,widx),axis=-1,dtype='float32') 
        
        del xidx,yidx,thidx,vidx,widx
        gc.collect()

        # give the self.current state and control, get the next state, without noise
        # in metric
        next_mean = self.car_next_state(utils.time_step, self.cell2metric(self.current[...,:3]), cell2control(self.current[...,3:],self.v_res,self.w_res), noise = False)
        # (10, 10, 10, 5, 10, 3)

        next_mean = np.tile(next_mean[np.newaxis,:,:,:,:,:,np.newaxis,:],\
                            (self.nt,1,1,1,1,1,self.nneighbors,1))
        # (100, 10, 10, 10, 5, 10, 6, 3)
        
        # find those whos out of the bound, renew their stage cost to inf
        # in cell
        # x_mask = (next_mean[...,0] < 0) | (next_mean[...,0] > 9)
        # y_mask = (next_mean[...,1] < 0) | (next_mean[...,1] > 9) 
        # in metric
        x_mask = (next_mean[...,0] <= -3) | (next_mean[...,0] >= 3)
        y_mask = (next_mean[...,1] <= -3) | (next_mean[...,1] >= 3) 

        mask = x_mask | y_mask
        self.l[mask] = np.inf
        del mask,x_mask,y_mask
        gc.collect()

        self.current = np.tile(self.current[np.newaxis,:,:,:,:,:,np.newaxis,:],\
                          (self.nt,1,1,1,1,1,self.nneighbors,1))
        # (100, 10, 10, 10, 5, 10, 6, 5)

        # in cell
        neighbors = self.metric2cell(next_mean[...,:3])
        # (100, 10, 10, 10, 5, 10, 6, 3)
        # next_mean: the next state after we apply the car_next_state function without noise
        #       but right now most of the state is not at exact cell spot
        #       this also the mean when it comes to the gaussian distribution
        
        add = np.array([[-1,0,0],
                        [1,0,0],
                        [0,-1,0],
                        [0,1,0],
                        [0,0,-1],
                        [0,0,1]]) 
        add = add[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:]
        add = np.tile(add,(self.nt,self.nx,self.ny,self.nth,self.nv,self.nw,1,1)) # in cell
        neighbors = neighbors + add  # in cell
        del add
        gc.collect()
        # (100, 10, 10, 10, 5, 10, 6, 3)
        x_mask = (neighbors[...,0] <= 0) | (neighbors[...,0] >= 9)
        y_mask = (neighbors[...,1] <= 0) | (neighbors[...,1] >= 9)  
        neighbor_mask = x_mask | y_mask
        # neighbors: the next, but we use np.floor to make all the state at the exact cell spot
        #            then we use add to find the neighbors

        # find the probability
        # dx = self.cell2metric(neighbors) - next_mean
        # self.P[...,3] = np.squeeze(self.get_probability(dx,next_mean,neighbor_mask)) # (100, 10, 10, 10, 5, 10, 6, 1)
        
        dx = neighbors - self.metric2cell(next_mean)
        self.P[...,3] = np.squeeze(self.get_probability(dx,self.metric2cell(next_mean),neighbor_mask)) # (100, 10, 10, 10, 5, 10, 6, 1)
        
        del next_mean,neighbors
        gc.collect()

        # in metric
        ref = self.ref_traj[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] # (nt,3)
        ref = np.tile(ref,(1,self.nx,self.ny,self.nth,self.nv,self.nw,self.nneighbors,1)) 
        # (100, 10, 10, 10, 5, 10, 6, 3)
        
        # get the next state with noise
        # in metric
        self.current[...,:3] = self.cell2metric(self.current[...,:3])
        self.current[...,3:] = cell2control(self.current[...,3:],self.v_res,self.w_res)
        self.P[...,:3] = self.current[...,:3] - ref


    def initPart2(self):
        
        for i in range(self.nt):
            self.current[i,...,:3] = self.car_next_state(utils.time_step, self.current[i,...,:3], self.current[i,...,3:],noise = False)
        # in metric
        self.l = self.find_cost(self.P[...,:3],self.current[...,3:])
        print(self.l.shape)
        

    def car_next_state(self, time_step, cur_state, control, noise=True):
        sigma = utils.sigma
        theta = cur_state[...,2] 
        zero = np.zeros_like(theta)
        one = np.ones_like(theta)
        temp1 = np.stack((np.cos(theta),zero),axis=-1) # (10, 10, 10, 5, 10, 2)
        temp2 = np.stack((np.sin(theta),zero),axis=-1)
        temp3 = np.stack((zero,one),axis=-1)
        rot_3d_z = np.stack((temp1,temp2,temp3),axis=-2) # (10, 10, 10, 5, 10, 3, 2)
    
        f = np.squeeze(rot_3d_z @ (control[...,np.newaxis])) # (10, 10, 10, 5, 10, 3)
        w_xy = np.random.normal(0, sigma[0], 2)
        w_theta = np.random.normal(0, sigma[2], 1)
        w = np.concatenate((w_xy, w_theta)) # (3,)

        if noise:
            return cur_state + time_step * f + w
        else:
            return cur_state + time_step * f

    def metric2cell(self,x):
        result = np.zeros_like(x)
        result[...,:2] =  np.floor((x[...,:2] - (-3))/self.x_res)
        result[...,2] = np.floor((x[...,2] - (-np.pi))/self.th_res)
        return result
    
    def cell2metric(self,x):
        result = np.zeros_like(x)
        result[...,:2] = (x[...,:2] + 0.5) * self.x_res + (-3)
        result[...,2] = (x[...,2] + 0.5) * self.th_res + (-np.pi)
        return result
          
    def get_probability(self,x,mean,mask,sigma = utils.sigma):

        k = x.shape[-1] # dimension of the state: 3
        var = np.diag(sigma**2)
        det = np.linalg.det(var)
        inv = np.linalg.inv(var)
        mult = 1/np.sqrt((2*np.pi)**k*det)
        
        # diff = (x-mean)[...,np.newaxis] # (100, 10, 10, 10, 5, 10, 6, 3, 1)
        diff = x[...,np.newaxis] # (100, 10, 10, 10, 5, 10, 6, 3, 1)
        diffT = np.moveaxis(diff,-1,-2) # (100, 10, 10, 10, 5, 10, 6, 1, 3)

        exp = np.squeeze((np.exp(-0.5 * diffT @ inv @ diff)),axis=-1)
        del diff,diffT
        gc.collect()
        prob = mult*exp #(100, 10, 10, 10, 5, 10, 6, 1)
        prob[mask[...,new]] = 0
        prob_mask = (np.abs(prob)>=0.00001)
        prob[~prob_mask] = 0
        # normalization
        temp = np.sum(prob,axis=-2,keepdims=True) # (100, 10, 10, 10, 5, 10, 1, 1)
        temp_mask = (np.abs(temp)>=0.00001)

        prob[prob_mask] = prob[prob_mask]/temp[temp_mask]
        return prob

    def find_cost(self,curr_e,control):
        """
        Aim: get the stage cost with given control and self.current error
        """
        q = 1
        Q = np.eye(2)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:]
        Q = np.tile(Q,(self.nt,self.nx,self.ny,self.nth,self.nv,self.nw,self.nneighbors,1,1))
        R = np.eye(2)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:]
        R = np.tile(R,(self.nt,self.nx,self.ny,self.nth,self.nv,self.nw,self.nneighbors,1,1))
        
        curr_x = curr_e[...,:2][...,np.newaxis]
        curr_xT = np.moveaxis(curr_x,-1,-2)
        curr_u = control[...,np.newaxis]
        curr_uT = np.moveaxis(curr_u,-1,-2)

        return np.squeeze(curr_xT @ Q @ curr_x +curr_uT @ R @ curr_u) + q*(1 - np.cos(curr_e[...,2]))**2
    
    def value_iteration(self,num_iter):
        # there is no terminal state
        gamma = 0.95
        print(gamma)

        # initialize the policy and value
        pi = np.zeros((self.nt,self.nx,self.ny,self.nth,2),dtype='float32')
        
        V = np.zeros((self.nt,self.nx,self.ny,self.nth,self.nv,self.nw,self.nneighbors),dtype='float32')
        # test code
        max_diff = np.zeros(num_iter)
        nChgActions = np.zeros(num_iter)
        newv = np.zeros_like(V)
        newpi = np.zeros_like(pi)
        

        for k in range(num_iter):
            Q = self.l + gamma * np.sum(self.P[...,3] * V, axis = -1)[...,new] # (100, 10, 10, 10, 5, 10, 6)
            # print(k)
            for t in range(self.nt):
                for x in range(self.nx):    
                    for y in range(self.ny):    
                        for th in range(self.nth):  
                            matrix = Q[t,x,y,th,:,:,1]
                            min_idx = np.unravel_index(np.argmin(matrix), matrix.shape)
                            newpi[t,x,y,th,:] = min_idx
                            newv[t,x,y,th,:,:,:] = np.min(matrix)
                            
            max_diff[k] = np.abs(newv - V).max()
            nChgActions[k] = (newpi != pi).sum()
            print("max diff:",max_diff[k])
            print("change action times:",nChgActions[k])
            V = newv
            pi = newpi  
            # test code
            # for t in range(self.nt):
            #     for x in range(self.nx):    
            #         for y in range(self.ny):    
            #             for th in range(self.nth):    
            #                 for v in range(self.nv):    
            #                     for w in range(self.nw):    
            #                         if np.all(Q[t,x,y,th,v,w] != Q[t,x,y,th,v,w,0]):
            #                             print("not equal")  
        
        self.V = V
        self.pi = pi
        return V, pi

def metric2cell(x,x_res,th_res):
    result = np.zeros_like(x)
    result[...,:2] =  np.floor((x[...,:2] - (-3))/x_res)
    result[...,2] = np.floor((x[...,2] - (-np.pi))/th_res)
    return result

def cell2metric(x,x_res,th_res):
    result = np.zeros_like(x)
    result[...,:2] = (x[...,:2] + 0.5) * x_res + (-3)
    result[...,2] = (x[...,2] + 0.5) * th_res + (-np.pi)
    return result

def cell2control(x,v_res,w_res):
    result = np.zeros_like(x)
    result[...,0] = (x[...,0] + 0.5) * v_res + (utils.v_min)
    result[...,1] = (x[...,1] + 0.5) * w_res + (utils.w_min)
    return result

def control2cell(x,v_res,w_res):  
    result = np.zeros_like(x)
    result[...,0] = (np.floor(x[...,0] - utils.v_min) / v_res)
    result[...,1] = (np.floor(x[...,1] - utils.w_min) / w_res)
    return result   

def find_traj(time_step,policy):
    x_res = (3-(-3))/10
    th_res = (np.pi-(-np.pi))/10
    v_res = (utils.v_max - utils.v_min)/5
    w_res = (utils.w_max - utils.w_min)/10
    init = np.array([utils.x_init,utils.y_init,utils.theta_init])
    traj = np.tile(init,(time_step,1))

    for i in range(time_step-1):

        x,y,th = metric2cell(traj[i],x_res,th_res)
        
        control = policy[i,int(x),int(y),int(th)]
        control = cell2control(control,v_res,w_res)
        traj[i+1] =  utils.car_next_state(utils.time_step, traj[i], control)
        if traj[i+1,2] >= np.pi:
            traj[i+1,2] = (traj[i+1,2] - np.pi) + (-np.pi)
        if traj[i+1,2] <=- np.pi:
            traj[i+1,2] = np.pi - (-np.pi - traj[i+1,2])

        if traj[i+1,0] >= 3:
            traj[i+1,0] = 2.95
        if traj[i+1,0] <= -3:
            traj[i+1,0] = -2.95

        if traj[i+1,1] >= 3:
            traj[i+1,1] = 2.95
        if traj[i+1,1] <= -3:
            traj[i+1,1] = -2.95
    
    return traj



def init():
    env = Environment(100)
    mdp = MDP(100,10,10,10,5,10,env.ref_traj)
    with open("mdp.pkl", "wb") as f:
        pickle.dump(mdp, f)

    return None 

def init2():
    with open("mdp.pkl", "rb") as f:
        mdp = pickle.load(f)
    mdp.initPart2()
    # save file with pickle
    with open("mdp.pkl", "wb") as f:
        pickle.dump(mdp, f)


def vi():
    with open("mdp.pkl", "rb") as f:
        mdp = pickle.load(f)
    V,pi = mdp.value_iteration(30)
    np.save('V.npy', V)
    np.save('pi.npy', pi)
    # save file with pickle
    with open("mdp.pkl", "wb") as f:
        pickle.dump(mdp, f)

def findTraj():
    V = np.load('V.npy')
    pi = np.load('pi.npy')
    
    # # 从文件加载对象
    with open("mdp.pkl", "rb") as f:
        mdp = pickle.load(f)
    # count = np.sum(pi > 10)
    # print(count)
    print("loading done")
    print(pi.shape)
    traj = find_traj(100,pi)
    obs = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
    utils.visualize(traj, mdp.ref_traj, obs, 100, utils.time_step, save=True)
    
    
    
if __name__ == "__main__":
    # init()
    # init2()
    # vi()
    findTraj()


        


