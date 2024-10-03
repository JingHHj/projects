import numpy as np
import modern_robotics as mr
import component1 as cp1
import component2 as cp2
import component3 as cp3
import component4 as cp4
import utils

def new_case():
    """
    cube in new case 
    robot have the default initial position 
    with initial error
    best tuned
    """
    cube_initial = (-np.pi/3,2,0)
    cube_goal = (0,1,-2)
    k = 3
    tau = 0.001
    Kp = np.eye(6) * 20.
    Ki = np.eye(6) * 0

    initial_state,Tse_initial = cp4.initial_error()
    states, X_error = cp4.component4(cube_initial,cube_goal,initial_state,Tse_initial,k,tau,Kp,Ki)
    
    name = 'new_case'
    utils.plot_error(X_error,tau,name)
    utils.get_csv_result(states,name)
    return None

def default_zero_init():
    """
    cube in defual setting 
    withou initial error 
    without any gains
    """
    cube_initial = (0,1,0)
    cube_goal = (-np.pi/2.,0,-1)
    k = 3
    tau = 0.001
    Kp = np.eye(6) * 0.
    Ki = np.eye(6) * 0.

    initial_state,Tse_initial = cp4.zero_initial_error()
    states, X_error = cp4.component4(cube_initial,cube_goal,initial_state,Tse_initial,k,tau,Kp,Ki)
    
    name = 'default_zero_init'
    utils.plot_error(X_error,tau,name)
    utils.get_csv_result(states,name)
    return None
    

def default_kp_only():
    """
    cube in defual setting 
    with initial error 
    with Kp gains
    """
    cube_initial = (0,1,0)
    cube_goal = (-np.pi/2.,0,-1)
    k = 3
    tau = 0.001
    Kp = np.eye(6) * 10.
    Ki = np.eye(6) * 0.

    initial_state,Tse_initial = cp4.initial_error()
    states, X_error = cp4.component4(cube_initial,cube_goal,initial_state,Tse_initial,k,tau,Kp,Ki)
    
    name = 'default_kp_only'
    utils.plot_error(X_error,tau,name)
    utils.get_csv_result(states,name)
    return None

def default_pi_control_overshoot():
    """
    cube in defual setting 
    with initial error 
    with Kp gains
    """
    cube_initial = (0,1,0)
    cube_goal = (-np.pi/2.,0,-1)
    k = 3
    tau = 0.001
    Kp = np.eye(6) * 8.
    Ki = np.eye(6) * 8.

    initial_state,Tse_initial = cp4.initial_error()
    states, X_error = cp4.component4(cube_initial,cube_goal,initial_state,Tse_initial,k,tau,Kp,Ki)
    
    name = 'default_pi_control_overshoot'
    utils.plot_error(X_error,tau,name)
    utils.get_csv_result(states,name)
    return None

def default_pi_control_oscillation():
    """
    cube in defual setting 
    with initial error 
    with Kp gains
    """
    cube_initial = (0,1,0)
    cube_goal = (-np.pi/2.,0,-1)
    k = 3
    tau = 0.001
    Kp = np.eye(6) * 4.
    Ki = np.eye(6) * 12.

    initial_state,Tse_initial = cp4.initial_error()
    states, X_error = cp4.component4(cube_initial,cube_goal,initial_state,Tse_initial,k,tau,Kp,Ki)
    
    name = 'default_pi_control_oscillation'
    utils.plot_error(X_error,tau,name)
    utils.get_csv_result(states,name)
    return None

def default_pi_best_tune():
    """
    cube in defual setting 
    with initial error 
    with Kp gains
    """
    cube_initial = (0,1,0)
    cube_goal = (-np.pi/2.,0,-1)
    k = 3
    tau = 0.001
    Kp = np.eye(6) * 15.
    Ki = np.eye(6) * 0.

    initial_state,Tse_initial = cp4.initial_error()
    states, X_error = cp4.component4(cube_initial,cube_goal,initial_state,Tse_initial,k,tau,Kp,Ki)
    
    name = 'default_pi_best_tune'
    utils.plot_error(X_error,tau,name)
    utils.get_csv_result(states,name)
    return None








def main():
    
    # new_case()
    
    # default_zero_init()
    
    # default_kp_only()
    
    # default_pi_control_oscillation()

    # default_pi_control_overshoot()
    
    # default_pi_best_tune()

    return None
 
if __name__ == '__main__':
   main()
