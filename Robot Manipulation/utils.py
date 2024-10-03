import numpy as np
import modern_robotics as mr
import pandas as pd
import matplotlib.pyplot as plt

def get_csv_result(result,name):
    """
    export result in csv file with comma in between every entrices
    """
    file = name + '.csv'
    df_result = pd.DataFrame(result)
    df_result.to_csv(file, index=False, sep=',')
    print("Successfully export " + name + "file")

    return None

def SE32output(traj,gripperstate):
    """
    put every trajectory with N SE(3) into the require output form, with 13 variables:
    r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state
    """
    traj = np.asarray(traj) # put the trajetory into ndarray so we can edit it easily
    result = np.zeros((traj.shape[0],13))
    for i,t in enumerate(traj):
        result[i] = np.hstack((t[:3,:3].flatten(),t[:3,3].flatten(),gripperstate))

    return result

def output2SE3(x):
    """
    put every trajectory with N SE(3) into the require output form, with 13 variables:
    r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state
    """
    T = np.array([
        [x[0],x[1],x[2],x[9]],
        [x[3],x[4],x[5],x[10]],
        [x[6],x[7],x[8],x[11]],
        [0,0,0,1]
    ])
    gripper = x[-1]

    return T,gripper


def BaseJacobian(M,Blist,thetalist):
    """
    Args:
        baselist (3,): (theta,x,y)
        Tse (4,4): The current actual end-effector configuration

    Returns:
        _type_: base jacobian
    """
    
    l = 0.47/2
    w = 0.3/2
    r = 0.0475
    F = r/4 * np.array([
        [-1./(l+w), 1./(l+w), 1./(l+w), -1./(l+w)],
        [       1.,       1.,       1.,        1.],
        [      -1.,       1.,      -1.,        1.]
    ])
    F6 = np.vstack((np.zeros((2,4)), F, np.zeros(4)))
    Tbo = np.array([
        [1,0,0,0.1662],
        [0,1,0,0],
        [0,0,1,0.0026],
        [0,0,0,1]
    ]) 
    Toe = mr.FKinBody(M,Blist,thetalist)
    matrix = mr.TransInv(Toe) @ mr.TransInv(Tbo)
    
    return mr.Adjoint(matrix) @ F6

def M_and_Blist():
    
    B1 = np.array([0,0,1,0,0.033,0])
    B2 = np.array([0,-1,0,-0.5076,0,0])
    B3 = np.array([0,-1,0,-0.3526,0,0])
    B4 = np.array([0,-1,0,-0.2176,0,0])
    B5 = np.array([0,0,1,0,0,0])
    Blist = np.vstack((B1,B2,B3,B4,B5), dtype=float).T
    
    M = np.array([
        [1,0,0,0.033],
        [0,1,0,0],
        [0,0,1,0.6546],
        [0,0,0,1]
    ]) 
    # M = np.eye(4,dtype=float)
    return M,Blist


def plot_error(twist_list,tau,name):
    y0 = twist_list[0]
    y1 = twist_list[1]
    y2 = twist_list[2]
    y3 = twist_list[3]
    y4 = twist_list[4]
    y5 = twist_list[5]
    
    x = np.arange(0,twist_list.shape[1]*tau,tau)
    # create plot
    plt.plot(x, y0, label='wx')
    plt.plot(x, y1, label='wy')
    plt.plot(x, y2, label='wz')
    plt.plot(x, y3, label='vx')
    plt.plot(x, y4, label='vy')
    plt.plot(x, y5, label='vz')
    # title and label
    plt.title(name)
    plt.xlabel('time')
    plt.ylabel('error')

    plt.legend()
    plt.grid(True)
    plt.savefig(name +'.png')
    plt.show()
    return None
 
def get_Tsb(phi,x,y):
    T = np.array([
        [np.cos(phi),-np.sin(phi),0,x],
        [np.sin(phi),np.cos(phi),0,y],
        [0,0,1,0.0963],
        [0,0,0,1]
    ])
    return T 

def get_Tbo():
    T = np.array([
        [1,0,0,0.1662],
        [0,1,0,0],
        [0,0,1,0.0026],
        [0,0,0,1]
    ])
    return T



def main():
    
   return None
 
if __name__ == '__main__':
   main()
