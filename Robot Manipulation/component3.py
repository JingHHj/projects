import numpy as np
import modern_robotics as mr
import utils

def FeedbackControl(config,X,Xd,Xd_next,Kp,Ki,tau,pre_error):
    """
    Inputs
        • config: current state of the robot (8 variables: 3 for chassis [:3], 5 for arm [3:]
        • The current actual end-effector configuration X (aka Tse)
        • The current reference end-effector configuration Xd (aka Tse,d)
        • The reference end-effector configuration at the next timestep, Xd,next (aka Tse,d,next)
        • The PI gain matrices Kp and Ki both (6,6)
        • The timestep ∆t between reference trajectory configurations
    Outputs
        • The commanded end-effector twist V expressed in the end-effector frame {e} (for plotting purposes).
        • The commanded wheel speeds, u and the commanded arm joint speeds, θ˙
    """
    Xerr = mr.TransInv(X) @ Xd
    Xerr_twist = mr.se3ToVec(mr.MatrixLog6(Xerr))
    Vd = mr.se3ToVec((1./tau) * mr.MatrixLog6(mr.TransInv(Xd) @ Xd_next))
    Adj_Xerr = mr.Adjoint(Xerr)

    pre_error = np.sum(pre_error,axis=1) + Xerr_twist
    Vb = Adj_Xerr @ Vd + Kp @ Xerr_twist + Ki @ pre_error * tau # not the best form
    
    # define Blist
    M,Blist = utils.M_and_Blist()
    thetalist = config[3:]
    J_arm = mr.JacobianBody(Blist,thetalist)
    J_base = utils.BaseJacobian(M,Blist,thetalist)
    
    Jacobian = np.hstack((J_base,J_arm))
    speeds = np.linalg.pinv(Jacobian) @ Vb
    
    # print information
    if __name__ == '__main__':
        print("Vd = ",Vd)
        print("Adj_Xerr @ Vd = ",Adj_Xerr @ Vd)
        print("The command twist V = ",Vb)
        print("Xerr = ",np.round(Xerr_twist,3))
        print("The increment to the numerical integral of the error should be \nXerr*deta_t = ",np.round(Xerr_twist*tau,3))
        print("The end_effector Jacobian \nJe=\n",np.round(Jacobian,3))
        print("The computation resutl \n(u, theta_dot) = ",np.round(speeds,1))
        
    return Vb,  speeds, Xerr_twist

def main():
    
    Xd = np.array([
        [0,0,1,0.5],
        [0,1,0,0],
        [-1,0,0,0.5],
        [0,0,0,1]
    ],dtype = float)
    
    Xd_next = np.array([
        [0,0,1,0.6],
        [0,1,0,0],
        [-1,0,0,0.3],
        [0,0,0,1]
    ],dtype = float)

    X = np.array([
        [0.170,0,0.985,0.387],
        [0,1,0,0],
        [-0.985,0,0.170,0.570],
        [0,0,0,1]
    ],dtype = float)

    Kp = Ki = np.zeros((6,6), dtype=float)
    # Kp = np.eye(6, dtype=float)
    t = 0.01
    config = np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0], dtype=float)
    Vb, speeds, Xerr_twist = FeedbackControl(config,X,Xd,Xd_next,Kp,Ki,t)
    return None


if __name__ == '__main__':
    main()








