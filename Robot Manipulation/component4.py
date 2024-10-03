import numpy as np
import modern_robotics as mr
import component1 as cp1
import component2 as cp2
import component3 as cp3
import utils

def zero_initial_error():
    """
    start with default setting ,with not zero error
    """
    M,Blist = utils.M_and_Blist()
        # phi,x,y,j1,j2,j3,j4,j5,w1,w2,w3,w4,w5,gripper
    initial_state = np.array([0,0,0,0,-1.57,0,0,0,0,0,0,0,0],dtype=float)
   
    Toe = mr.FKinBody(M,Blist,initial_state[3:8])
    Tsb = utils.get_Tsb(initial_state[0],initial_state[1],initial_state[2])
    Tbo = utils.get_Tbo()
    Tse_initial = Tsb @ Tbo @ Toe

    return initial_state,Tse_initial

def initial_error():
    """
    start with default setting but not zero error
    """
        # phi,x,y,j1,j2,j3,j4,j5,w1,w2,w3,w4,w5,gripper
    initial_state = np.array([np.pi/2,0.5,0,0,0,0,0,0,0,0,0,0,0],dtype=float)
    Tse_initial = np.array([
        [0,0,1,0],
        [0,1,0,0],
        [-1,0,0,0.5],
        [0,0,0,1]
    ])
    
    return initial_state,Tse_initial


def cube(cube_0,cube_1):
    
    phi_0,x0,y0 = cube_0
    phi_1,x1,y1 = cube_1
    
    Tsc_initial = np.array([
        [np.cos(phi_0),-np.sin(phi_0),0,x0],
        [np.sin(phi_0),np.cos(phi_0),0,y0],
        [0.,0.,1.,0.025],
        [0.,0.,0.,1.]
    ])  
    Tsc_final = np.array([
        [np.cos(phi_1),-np.sin(phi_1),0,x1],
        [np.sin(phi_1),np.cos(phi_1),0,y1],
        [ 0.,0.,1., 0.025],
        [ 0.,0.,0., 1.]
    ])  
    Tce_grasp = np.array([
        [ 0.,0.,1.,0.043-0.025],
        [ 0.,1.,0.,0.],
        [-1.,0.,0.,0],
        [ 0.,0.,0.,1.]
    ])
    Tce_standoff = np.array([
        [ 0.,0.,1.,0.043-0.025],
        [ 0.,1.,0.,0.],
        [-1.,0.,0.,0.2-0.025],
        [ 0.,0.,0.,1.]
    ])
    
    return  Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff



def component4(cube_initial,cube_goal,initial_state,Tse_initial,k,tau,Kp,Ki):
    

    Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff = cube(cube_initial,cube_goal)
    trajecory_SE3, trajecory_output = cp2.TrajectoryGenerator(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k)
    N = trajecory_SE3.shape[0] # number of reference pose in this trajectory
    
    states = np.empty((N,initial_state.shape[0])) # states list
    states[0] = initial_state # initial_state
    states[:,-1] = trajecory_output[:,-1]
    M,Blist = utils.M_and_Blist()
    X_error = np.zeros((6,N-1)) # error list
    Tbo = utils.get_Tbo()
    
    for i in range(N-1):
        Toe = mr.FKinBody(M,Blist,states[i,3:8])
        Tsb = utils.get_Tsb(states[i,0],states[i,1],states[i,2])
        X = Tsb @ Tbo @ Toe
        twist_i, speeds_i, X_error[:,i] = cp3.FeedbackControl( states[i,:8],X,trajecory_SE3[i,:,:],trajecory_SE3[i+1,:,:],Kp,Ki,tau,X_error)
        states[i+1,:12] = cp1.NextState(states[i,:12],speeds_i,tau,100000000)
        # print(i)
    
    return states, X_error
    
def main():
   
    return None
 
if __name__ == '__main__':
   main()
