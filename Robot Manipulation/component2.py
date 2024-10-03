import modern_robotics as mr
import numpy as np
import utils

def TrajectoryGenerator(Tse_inital, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k):
    """
    Inputs
        • The initial configuration of the end-effector: 
            Tse,initial
        • The initial configuration of the cube: 
            Tsc,initial
        • The desired final configuration of the cube: 
            Tsc,final
        • The configuration of the end-effector relative to the cube while grasping: 
            Tce,close
        • The standoff configuration of the end-effector above the cube, before and after grasping, relative to the cube:
            Tce,standoff
        • The number of trajectory reference configurations per 0.01 seconds: 
            k. (The value k is an integer with a value of 1 or greater. )
    Outputs
        • A representation of the N configurations of the end-effector along the entire concatenated eight-segment reference trajectory. 
        Each of these N reference points represents a transformation matrix Tse of the end-effector
        frame {e} relative to {s} at an instant in time, plus the gripper state (0 for open or 1 for closed). These
        reference configurations will be used by your controller. Note: if your trajectory takes t seconds, your function
        should generate N = Tf · k/0.01 configurations.
        • A .csv file with the entire eight-segment reference trajectory. Each line of the .csv file corresponds to one
        configuration Tse of the end-effector, expressed as 13 variables separated by commas. The 13 variables are, in
        order:
        r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state
    """
    Tf = 2
    n = Tf * k/0.01
    N_basic = n
    N_change = n # N for the grasping and releasing trajectory

    method = 5
        # define the state variables for the end effector to open or close 
    close = 1.
    open = 0.
        # computing every trajectory
    
    traj1_SE3 = np.copy(mr.CartesianTrajectory(Tse_inital, Tsc_initial@Tce_standoff, Tf, N_basic, method)) # 0
    traj2_SE3 = np.copy(mr.CartesianTrajectory(Tsc_initial@Tce_standoff, Tsc_initial@Tce_grasp, Tf, N_basic, method))
    traj3_SE3 = np.copy(mr.CartesianTrajectory(Tsc_initial@Tce_grasp, Tsc_initial@Tce_grasp, Tf, N_change, method))
    # the close proccess
    traj4_SE3 = np.copy(mr.CartesianTrajectory(Tsc_initial@Tce_grasp, Tsc_initial@Tce_standoff, Tf, N_basic, method)) # 1
    traj5_SE3 = np.copy(mr.CartesianTrajectory(Tsc_initial@Tce_standoff, Tsc_final@Tce_standoff, Tf, N_basic, method)) # 1
    traj6_SE3 = np.copy(mr.CartesianTrajectory(Tsc_final@Tce_standoff, Tsc_final@Tce_grasp, Tf, N_basic, method)) # 1
    traj7_SE3 = np.copy(mr.CartesianTrajectory(Tsc_final@Tce_grasp, Tsc_final@Tce_grasp, Tf, N_change, method)) # 0
    # the open process
    traj8_SE3 = np.copy(mr.CartesianTrajectory(Tsc_final@Tce_grasp, Tsc_final@Tce_standoff, Tf, N_basic, method)) # 0
    total_traj_SE3 = np.vstack((traj1_SE3, traj2_SE3, traj3_SE3,traj4_SE3, traj5_SE3, traj6_SE3, traj7_SE3, traj8_SE3))

    traj1_output = utils.SE32output(traj1_SE3,0) # 0
    traj2_output = utils.SE32output(traj2_SE3,0) # 0
    traj3_output = utils.SE32output(traj3_SE3,1) # 1
    traj4_output = utils.SE32output(traj4_SE3,1) # 1
    traj5_output = utils.SE32output(traj5_SE3,1) # 1
    traj6_output = utils.SE32output(traj6_SE3,1) # 1
    traj7_output = utils.SE32output(traj7_SE3,0) # 0
    traj8_output = utils.SE32output(traj8_SE3,0) # 0
    total_traj_output = np.vstack((traj1_output, traj2_output, traj3_output,traj4_output, traj5_output, traj6_output, traj7_output, traj8_output))

    return total_traj_SE3, total_traj_output

def DefaultCube():
    Tsc_initial = np.array([
        [1.,0.,0.,1.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.025],
        [0.,0.,0.,1.]
    ])  # (x,y,z) = (1,0,0)
    Tsc_final = np.array([
        [ 0.,1.,0., 0.],
        [-1.,0.,0.,-1.],
        [ 0.,0.,1., 0.025],
        [ 0.,0.,0., 1.]
    ])  # (x,y,z) = (0, −1, −π/2)
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

    return Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff
    

def component2():
     # z=0.0963 meters is the height of the {b} frame above the floor.
    l = 0.2761 + 0.1350 + 0.1550 + 0.1470 # the length of the whole robot arm
    Tse_inital = np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,l],
        [0.,0.,0.,1.]
    ])
    Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff = DefaultCube()
    
    k = 3
    traj_SE3, traj_output = TrajectoryGenerator(Tse_inital, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k)
    return traj_SE3, traj_output
    

def main():
    trajecory_SE3, trajecory_output = component2()
    name = 'component2.csv'
    utils.get_csv_result(trajecory_output,name)
    return None
 
if __name__ == '__main__':
   main()



