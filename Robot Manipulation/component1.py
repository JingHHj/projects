import modern_robotics as mr
import numpy as np
import pandas as pd
import utils

def NextState(current_state,velocities,timestep,max):
    """
    Inputs
        • The current state of the robot 
            (12 variables: 3 for chassis [:3], 5 for arm [3:8], 4 for wheel angles [8:])
        • The joint and wheel velocities 
            (9 variables: 5 for arm θ˙ [4:], 4 for wheels u [:4])
        • The timestep size ∆t (1 parameter)
        • The maximum joint and wheel velocity magnitude (1 parameter)
    return:
        NextState should also produce the following outputs that describe the configuration of the robot one
        timestep (∆t) later:
        • The next state (configuration) of the robot (12 variables)
    """
    # if np.any(velocities > max):
    #     print("The given speed is bigger than the maximum")
    #     return None  # if there's velocity that is bigger than the maximum, then quit the function
    # else:
            
    
    l = 0.47/2
    w = 0.3/2
    r = 0.0475
    new_state = np.copy(current_state)

    
    wheel_v = velocities[:4]
    joint_v = velocities[4:]
    # joints
    new_state[3:8] = current_state[3:8] + joint_v*timestep
    # wheels
    new_state[8:12] = current_state[8:12] + wheel_v*timestep
    # chassis
    F = r/4 *np.array([
        
        [-1./(l+w), 1./(l+w), 1./(l+w), -1./(l+w)],
        [       1.,       1.,       1.,        1.],
        [      -1.,       1.,      -1.,        1.]
    ])
    vb =  F @ (wheel_v*timestep)
    wz,vx,vy = vb
    
    if wz == 0. : 
        delta_qb = np.array([
            0, vx, vy
            ])   
    else:
        delta_qb = np.array([
            wz,
            (vx * np.sin(wz) + vy * (np.cos(wz)-1))/wz,
            (vy * np.sin(wz) + vx * (1-np.cos(wz)))/wz
        ])
    R0 = np.array([
        [1,0,0],
        [0,np.cos(current_state[0]),-np.sin(current_state[0])],
        [0,np.sin(current_state[0]),np.cos(current_state[0])]
                ])
    
    delta_q = R0 @ delta_qb 
    new_state[:3] = current_state[:3] + delta_q
        
    return new_state

def component1(initial_state,test_v,timestep):
    
    maximum = 5000.
    iteration_step = 100
    i = 0
    state_list = np.zeros((iteration_step+1,13)) # plus a “0” for “gripper open”
    state_list[0,:12] = initial_state
    while i < iteration_step:
        state_list[i+1,:12] = NextState(state_list[i,:12],test_v,timestep,maximum)
        i = i +1

    return state_list


def main():
    # initializing things we need for cmoputation
    initial_state = np.array([0,0,0,0,0,0,0,0,0,0,0,0],dtype=float)
    
    test_v = np.array([-10.,10.,10.,-10.,0.,0.,0.,0.,0.])
    timestep = 0.01
    states =component1(initial_state,test_v,timestep)
    name = 'component1.csv'
    utils.get_csv_result(states,name)
    return None
 
if __name__ == '__main__':
   main()



    



